from pathlib import Path
from collections import deque, OrderedDict
import networkx as nx

from loki.frontend import FP, REGEX, RegexParserClass
from loki.sourcefile import Sourcefile
from loki.dimension import Dimension
from loki.tools import as_tuple, CaseInsensitiveDict, timeit
from loki.logging import info, warning, debug, PERF
from loki.bulk.item import ProcedureBindingItem, SubroutineItem


__all__ = ['Scheduler', 'SchedulerConfig']


class SchedulerConfig:
    """
    Configuration object for the transformation :any:`Scheduler` that
    encapsulates default behaviour and item-specific behaviour. Can
    be create either from a raw dictionary or configration file.

    Parameters
    ----------
    default : dict
        Default options for each item
    routines : dict of dicts or list of dicts
        Dicts with routine-specific options.
    dimensions : dict of dicts or list of dicts
        Dicts with options to define :any`Dimension` objects.
    disable : list of str
        Subroutine names that are entirely disabled and will not be
        added to either the callgraph that we traverse, nor the
        visualisation. These are intended for utility routines that
        pop up in many routines but can be ignored in terms of program
        control flow, like ``flush`` or ``abort``.
    """

    def __init__(self, default, routines, disable=None, dimensions=None):
        self.default = default
        if isinstance(routines, dict):
            self.routines = CaseInsensitiveDict(routines)
        else:
            self.routines = CaseInsensitiveDict((r.name, r) for r in as_tuple(routines))
        self.disable = as_tuple(disable)
        self.dimensions = dimensions

    @classmethod
    def from_dict(cls, config):
        default = config['default']
        if 'routine' in config:
            config['routines'] = OrderedDict((r['name'], r) for r in config.get('routine', []))
        else:
            config['routines'] = []
        routines = config['routines']
        disable = default.get('disable', None)

        # Add any dimension definitions contained in the config dict
        dimensions = {}
        if 'dimension' in config:
            dimensions = [Dimension(**d) for d in config['dimension']]
            dimensions = {d.name: d for d in dimensions}

        return cls(default=default, routines=routines, disable=disable, dimensions=dimensions)

    @classmethod
    def from_file(cls, path):
        import toml  # pylint: disable=import-outside-toplevel
        # Load configuration file and process options
        with Path(path).open('r') as f:
            config = toml.load(f)

        return cls.from_dict(config)


class Scheduler:
    """
    Work queue manager to enqueue and process individual `Item`
    routines/modules with a given kernel.

    Note: The processing module can create a callgraph and perform
    automated discovery, to enable easy bulk-processing of large
    numbers of source files.

    Parameters
    ----------
    paths : str or list of str
        List of paths to search for automated source file detection.
    config : dict or str, optional
        Configuration dict or path to scheduler configuration file
    preprocess : bool, optional
        Flag to trigger CPP preprocessing (by default `False`).
    includes : list of str, optional
        Include paths to pass to the C-preprocessor.
    defines : list of str, optional
        Symbol definitions to pass to the C-preprocessor.
    definitions : list of :any:`Module`, optional
        :any:`Module` object(s) that may supply external type or procedure
        definitions.
    xmods : str, optional
        Path to directory to find and store ``.xmod`` files when using
        the OMNI frontend.
    omni_includes: list of str, optional
        Additional include paths to pass to the preprocessor run as part of
        the OMNI frontend parse. If set, this **replaces** (!)
        :data:`includes`, otherwise :data:`omni_includes` defaults to the
        value of :data:`includes`.
    frontend : :any:`Frontend`, optional
        Frontend to use when parsing source files (default :any:`FP`).
    """

    # TODO: Should be user-definable!
    source_suffixes = ['.f90', '.F90', '.f', '.F']

    def __init__(self, paths, config=None, preprocess=False, includes=None,
                 defines=None, definitions=None, xmods=None, omni_includes=None,
                 frontend=FP):
        # Derive config from file or dict
        if isinstance(config, SchedulerConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = SchedulerConfig.from_file(config)
        else:
            self.config = SchedulerConfig.from_dict(config)

        # Build-related arguments to pass to the sources
        self.paths = [Path(p) for p in as_tuple(paths)]

        # Accumulate all build arguments to pass to `Sourcefile` constructors
        self.build_args = {
            'definitions': definitions,
            'preprocess': preprocess,
            'includes': includes,
            'defines': defines,
            'xmods': xmods,
            'omni_includes': omni_includes,
            'frontend': frontend
        }

        # Internal data structures to store the callgraph
        self.item_graph = nx.DiGraph()
        self.item_map = {}

        self._discover()

    @timeit(log_level=PERF)
    def _discover(self):
        # Scan all source paths and create light-weight `Sourcefile` objects for each file.
        frontend_args = {
            'preprocess': self.build_args['preprocess'],
            'includes': self.build_args['includes'],
            'defines': self.build_args['defines'],
            'parser_classes': RegexParserClass.All,
            'frontend': REGEX
        }

        obj_list = []
        for path in self.paths:
            for ext in self.source_suffixes:
                obj_list += [Sourcefile.from_file(filename=f, **frontend_args) for f in path.glob(f'**/*{ext}')]

        debug(f'Total number of lines parsed: {sum(obj.source.lines[1] for obj in obj_list)}')

        # Create a map of all potential target objs for fast lookup later
        self.obj_map = CaseInsensitiveDict(
            (f'#{r.name}', obj) for obj in obj_list for r in obj.subroutines
        )
        self.obj_map.update(
            (f'{module.name}#{r.name}', obj)
            for obj in obj_list for module in obj.modules
            for r in module.subroutines + tuple(module.typedefs.values())
        )

    @property
    def routines(self):
        return as_tuple(item.routine for item in self.item_graph.nodes if item.routine is not None)

    @property
    def items(self):
        """
        All :any:`Item` objects contained in the :any:`Scheduler` call graph.
        """
        return as_tuple(self.item_graph.nodes)

    @property
    def dependencies(self):
        """
        All individual pairs of :any:`Item` that represent a dependency
        and form an edge in the :any`Scheduler` call graph.
        """
        return as_tuple(self.item_graph.edges)

    def create_item(self, name):
        """
        Create an `Item` by looking up the path and setting all inferred properties.

        If the item cannot be created due to unknown source files, and the default
        configuration does not force ``strict`` behaviour, ``None`` is returned.

        Note that this takes a `SchedulerConfig` object for default options and an
        item-specific dict with override options, as well as given attributes that
        might be forced on this item from its parent.
        """
        if name in self.item_map:
            return self.item_map[name]

        if isinstance(name, tuple):
            # If we have a list of candidates, try to create the item for each and
            # use the first one that is successful
            for candidate in name:
                item = None
                try:
                    item = self.create_item(candidate)
                except FileNotFoundError:
                    continue
                if item is not None:
                    break
            if item is None and self.config.default['strict']:
                raise FileNotFoundError(f'Sourcefile not found for {name}')
            return item

        name = name.lower()

        # For type bound procedures, we use the (fully-qualified) type name to
        # look up the corresponding sourcefile (since we're not storing every
        # procedure binding in obj_map)
        pos = name.find('%')
        if pos == -1:
            sourcefile = self.obj_map.get(name)
        else:
            sourcefile = self.obj_map.get(name[:pos])

        if sourcefile is None:
            warning(f'Scheduler could not create item: {name}')
            if self.config.default['strict']:
                raise FileNotFoundError(f'Sourcefile not found for {name}')
            return None

        # Use default as base and override individual options
        item_conf = self.config.default.copy()
        routine_names = {name, name[name.index('#')+1:]}
        config_matches = [r for r in self.config.routines if r.lower() in routine_names]
        if config_matches:
            if len(config_matches) != 1:
                warning(f'Multiple config entries matching {name}')
                if self.config.default['strict']:
                    raise RuntimeError(f'Multiple config entries matching {name}')
            item_conf.update(self.config.routines[config_matches[0]])

        debug(f'[Loki] Scheduler creating Item: {name} => {sourcefile.path}')
        if '%' in name:
            return ProcedureBindingItem(name=name, source=sourcefile, config=item_conf)
        return SubroutineItem(name=name, source=sourcefile, config=item_conf)

    def find_routine(self, routine):
        """
        Find the given routine name in the :attr:`obj_map`

        This looks for matching candidates, possibly ignoring module scopes.
        If this yields more than one match, it will print a warning and use the
        first match. If ``strict`` mode is active, :class:`RuntimeError` is raised.

        This is used when filling the scheduler graph with the initial starting
        points without the need to provide these with fully-qualified names.

        Parameters
        ----------
        routine : str
            The name of the routine to look for

        Returns
        -------
        str :
            The fully-qualified name corresponding to :data:`routine` from
            the set of discovered routines
        """
        if '#' in routine:
            name = routine.lower()
        else:
            name = f'#{routine.lower()}'

        candidates = [c for c in self.obj_map if c.lower().endswith(name)]
        if not candidates:
            warning(f'Scheduler could not find routine {routine}')
            if self.config.default['strict']:
                raise RuntimeError(f'Scheduler could not find routine {routine}')
        elif len(candidates) != 1:
            warning(f'Scheduler found multiple candidates for routine {routine}: {candidates}')
            if self.config.default['strict']:
                raise RuntimeError(f'Scheduler found multiple candidates for routine {routine}: {candidates}')
        return candidates[0]

    @timeit(log_level=PERF)
    def populate(self, routines):
        """
        Populate the callgraph of this scheduler through automatic expansion of
        subroutine-call induced dependencies from a set of starting routines.

        Parameters
        ----------
        routines : list of str
            Names of root routines from which to populate the callgraph.
        """
        queue = deque()
        for routine in as_tuple(routines):
            item = self.create_item(self.find_routine(routine))
            if item:
                queue.append(item)

                self.item_map[item.name] = item
                self.item_graph.add_node(item)

        while len(queue) > 0:
            item = queue.popleft()

            children = item.qualify_names(item.children, available_names=self.obj_map.keys())
            for c in children:
                child = self.create_item(c)

                if child is None:
                    continue

                # Skip blocked children as well
                if child.local_name in item.block:
                    continue

                # Append child to work queue if expansion is configured
                if item.expand:
                    # Do not propagate to dependencies marked as "ignore"
                    # Note that, unlike blackisted items, "ignore" items
                    # are still marked as targets during bulk-processing,
                    # so that calls to "ignore" routines will be renamed.
                    if child.local_name in item.ignore:
                        continue

                    if child not in self.item_map:
                        queue.append(child)
                        self.item_map[child.name] = child
                        self.item_graph.add_node(child)

                    self.item_graph.add_edge(item, child)

    @timeit(log_level=PERF)
    def enrich(self):
        """
        Enrich subroutine calls for inter-procedural transformations
        """
        # Force the parsing of the routines in the call tree
        for item in self.item_graph:
            item.source.make_complete(**self.build_args)

        for item in self.item_graph:
            if not isinstance(item, SubroutineItem):
                continue

            # Enrich with all routines in the call tree
            item.routine.enrich_calls(routines=self.routines)

            # Enrich item with meta-info from outside of the callgraph
            for routine in item.enrich:
                lookup_name = self.find_routine(routine)
                if not lookup_name:
                    warning(f'Scheduler could not find file for enrichment:\n{routine}')
                    if self.config.default['strict']:
                        raise FileNotFoundError(f'Source path not found for routine {routine}')
                    continue
                self.obj_map[lookup_name].make_complete(**self.build_args)
                item.routine.enrich_calls(self.obj_map[lookup_name].all_subroutines)

    @timeit(log_level=PERF, getter=lambda x: str(x.get('transformation', '')))
    def process(self, transformation, reverse=False):
        """
        Process all enqueued source modules and routines with the
        stored kernel. The traversal is performed in topological
        order, which ensures that :any:`CallStatement` objects are
        always processed before their target :any:`Subroutine`.
        """
        # Enrich routines in graph with type info
        self.enrich()

        traversal = nx.topological_sort(self.item_graph)
        if reverse:
            traversal = reversed(list(traversal))

        for item in traversal:
            if not isinstance(item, SubroutineItem):
                continue

            # Process work item with appropriate kernel
            transformation.apply(item.source, role=item.role, mode=item.mode,
                                 item=item, targets=item.targets)

    def callgraph(self, path):
        """
        Generate a callgraph visualization and dump to file.

        :param path: Path to write the callgraph figure to.
        """
        try:
            import graphviz as gviz  # pylint: disable=import-outside-toplevel
        except ImportError:
            warning('[Loki] Failed to load graphviz, skipping callgraph generation...')
            return

        cg_path = Path(path)
        callgraph = gviz.Digraph(format='pdf', strict=True, graph_attr=(('rankdir', 'LR'),))

        # Insert all nodes in the schedulers graph
        for item in self.items:
            if item.replicate:
                callgraph.node(item.name.upper(), color='black', shape='diamond',
                                fillcolor='limegreen', style='rounded,filled')
            else:
                callgraph.node(item.name.upper(), color='black', shape='box',
                                fillcolor='limegreen', style='filled')

        # Insert all edges in the schedulers graph
        for parent, child in self.dependencies:
            callgraph.edge(parent.name.upper(), child.name.upper())  # pylint: disable=no-member

        # Insert all nodes we were told to either block or ignore
        for item in self.items:
            blocked_children = [child for child in item.children if child in item.block]
            blocked_children = item.qualify_names(blocked_children, self.obj_map.keys())
            blocked_children = [child for child in blocked_children if isinstance(child, str)]
            for child in blocked_children:
                callgraph.node(child.upper(), color='black', shape='box',
                            fillcolor='orangered', style='filled')
                callgraph.edge(item.name.upper(), child.upper())

            ignored_children = [child for child in item.children if child in item.ignore]
            ignored_children = item.qualify_names(ignored_children, self.obj_map.keys())
            ignored_children = [child for child in ignored_children if isinstance(child, str)]
            for child in ignored_children:
                callgraph.node(child.upper(), color='black', shape='box',
                            fillcolor='lightblue', style='filled')
                callgraph.edge(item.name.upper(), child.upper())

            missing_children = item.qualify_names(item.children, self.obj_map.keys())
            missing_children = [child[0] for child in missing_children if isinstance(child, tuple)]
            for child in missing_children:
                callgraph.node(child.upper(), color='black', shape='box',
                            fillcolor='lightgray', style='filled')
                callgraph.edge(item.name.upper(), child.upper())

        try:
            callgraph.render(cg_path, view=False)
        except gviz.ExecutableNotFound as e:
            warning(f'[Loki] Failed to render callgraph due to graphviz error:\n  {e}')

    @timeit(log_level=PERF)
    def write_cmake_plan(self, filepath, mode, buildpath, rootpath):
        """
        Generate the "plan file", a CMake file defining three lists
        that contain the respective files to append / remove /
        transform. These lists are used by the CMake wrappers to
        schedule the source updates and update the source lists of the
        CMake target object accordingly.
        """
        info(f'[Loki] Scheduler writing CMake plan: {filepath}')

        rootpath = Path(rootpath).resolve()
        buildpath = None if buildpath is None else Path(buildpath)
        sources_to_append = []
        sources_to_remove = []
        sources_to_transform = []

        for item in self.items:
            sourcepath = item.path.resolve()
            newsource = sourcepath.with_suffix(f'.{mode.lower()}.F90')
            if buildpath:
                newsource = buildpath/newsource.name

            # Make new CMake paths relative to source again
            sourcepath = sourcepath.relative_to(rootpath)

            debug(f'Planning:: {item.name} (role={item.role}, mode={mode})')

            sources_to_transform += [sourcepath]

            # Inject new object into the final binary libs
            if item.replicate:
                # Add new source file next to the old one
                sources_to_append += [newsource]
            else:
                # Replace old source file to avoid ghosting
                sources_to_append += [newsource]
                sources_to_remove += [sourcepath]

        info(f'[Loki] CMakePlanner writing plan: {filepath}')
        with Path(filepath).open('w') as f:
            s_transform = '\n'.join(f'    {s}' for s in sources_to_transform)
            f.write(f'set( LOKI_SOURCES_TO_TRANSFORM \n{s_transform}\n   )\n')

            s_append = '\n'.join(f'    {s}' for s in sources_to_append)
            f.write(f'set( LOKI_SOURCES_TO_APPEND \n{s_append}\n   )\n')

            s_remove = '\n'.join(f'    {s}' for s in sources_to_remove)
            f.write(f'set( LOKI_SOURCES_TO_REMOVE \n{s_remove}\n   )\n')
