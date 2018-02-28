from ecir.visitors import Visitor
from ecir.tools import chunks

__all__ = ['fgen', 'FortranCodegen']


class FortranCodegen(Visitor):
    """
    Tree visitor to generate standardized Fortran code from IR.
    """

    def __init__(self, depth=0, chunking=6, conservative=True):
        super(FortranCodegen, self).__init__()
        self.conservative = conservative
        self.chunking = chunking
        self._depth = 0

    @classmethod
    def default_retval(cls):
        return ""

    @property
    def indent(self):
        return '  ' * self._depth

    def segment(self, arguments):
        delim = ', &\n%s & ' % self.indent
        args = list(chunks([self.visit(a) for a in arguments], self.chunking))
        return '%s & ' % self.indent + delim.join(', '.join(c) for c in args)

    def visit(self, o):
        if self.conservative and hasattr(o, '_source') and o._source is not None:
            # Re-use original source associated with node
            return o._source
        else:
            return super(FortranCodegen, self).visit(o)

    def visit_Node(self, o):
        return self.indent + '! <%s>' % o.__class__.__name__

    def visit_tuple(self, o):
        return '\n'.join([self.visit(i) for i in o])

    visit_list = visit_tuple

    def visit_Comment(self, o):
        return self.indent + o._source

    def visit_CommentBlock(self, o):
        comments = [self.visit(c) for c in o.comments]
        return '\n'.join(comments)

    def visit_Declaration(self, o):
        type = self.visit(o.variables[0].type)
        return self.indent + '%s :: %s' % (type, ', '.join(str(v) for v in o.variables))

    def visit_Import(self, o):
        return 'USE %s, ONLY: %s' % (o.module, ', '.join(o.symbols))

    def visit_Loop(self, o):
        self._depth += 1
        body = self.visit(o.children)
        self._depth -= 1
        header = '%s=%s, %s' % (o.variable, o.bounds[0], o.bounds[1])
        return self.indent + 'DO %s\n%s\n%sEND DO' % (header, body, self.indent)

    def visit_Call(self, o):
        if len(o.arguments) > 6:
            self._depth += 2
            arguments = self.segment(o.arguments)
            self._depth -= 2
            signature = 'CALL %s( &\n%s )' % (o.name, arguments)
        else:
            signature = 'CALL %s(%s)' % (o.name, ', '.join(o.arguments))
        return self.indent + signature

    def visit_Variable(self, o):
        dims = '(%s)' % ','.join([str(d) for d in o.dimensions]) if len(o.dimensions) > 0 else ''
        return '%s%s' % (o.name, dims)

    def visit_Type(self, o):
        return '%s%s%s%s%s%s' % (o.name, '(KIND=%s)' % o.kind if o.kind else '',
                                 ', INTENT(%s)' % o.intent.upper() if o.intent else '',
                                 ', ALLOCATE' if o.allocatable else '',
                                 ', POINTER' if o.pointer else '',
                                 ', OPTIONAL' if o.optional else '')

def fgen(ir, depth=0, chunking=6, conservative=True):
    """
    Generate standardized Fortran code from one or many IR objects/trees.
    """
    return FortranCodegen(depth=depth, chunking=chunking,
                          conservative=conservative).visit(ir)
