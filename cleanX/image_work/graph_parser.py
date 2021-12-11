# -*- coding: utf-8 -*-

import string

from .steps import get_known_steps

class Parser:

    name_chars = string.ascii_letters + string.digits

    def __init__(self, stream):
        self.stream = stream
        self.ctors = get_known_steps()
        self.overflow = None
        # TODO(olegs): Track position in the parser
        self.line = 0
        self.column = 0

    def peek(self):
        if self.overflow is None:
            self.overflow = self.stream.read(1)
            if not self.overflow:
                raise EOFError()
        return self.overflow

    def consume(self):
        self.overflow = None

    def read_blank(self):
        while True:
            c = self.peek()
            if c not in string.whitespace:
                break
            self.consume()

    def raise_syntax_error(self, c):
        err = SyntaxError('Unexpected {!r}'.format(c))
        err.lineno = self.line
        err.offset = self.column
        raise err

    def read(self, cs):
        for c in cs:
            d = self.peek()
            if c != d:
                self.raise_syntax_error(d)
            self.consume()

    def read_arrow(self):
        self.read_blank()
        self.read('-')
        self.read('>')

    def read_step(self):
        raw = []
        while True:
            c = self.peek()
            if c in ('\n', '#'):
                break
            raw.append(c)
            self.consume()
        l, g = locals(), globals()
        l = dict(l)
        l.update(self.ctors)
        return eval(''.join(raw), l, g)

    def read_label(self):
        raw = []
        while True:
            c = self.peek()
            if c == ':':
                self.consume()
                break
            raw.append(c)
            self.consume()
        if not raw:
            self.raise_syntax_error('empty label')
        return ''.join(raw)

    def read_comment(self):
        self.read_blank()
        if self.peek() != '#':
            return False
        self.consume()

        while self.peek() != '\n':
            self.consume()
        return True

    def read_comments(self):
        while self.read_comment():
            pass

    def read_vertices(self):
        self.read_blank()
        self.read_comments()
        self.read_blank()
        self.read('vertices:')
        self.read_blank()
        self.read_comments()

        result = {}

        while True:
            label = self.read_label()
            if label == 'arcs':
                break
            if label in result:
                self.raise_syntax_error(
                    'duplicate label: {}'.format(label),
                )
            result[label] = self.read_step()
            self.read_comments()

        return result

    def read_word(self):
        result = []

        while True:
            c = self.peek()
            if c in string.whitespace:
                if not result:
                    self.raise_syntax_error('mising word')
                break
            result.append(c)
            self.consume()
        return ''.join(result)

    def read_arcs(self, labels):
        self.read_blank()
        self.read_comments()

        while True:
            self.read_blank()
            a = self.read_word()
            if a not in labels:
                self.raise_syntax_error(a)
            self.read_blank()
            self.read_arrow()
            self.read_blank()
            b = self.read_word()
            if b not in labels:
                self.raise_syntax_error(b)
            yield a, b
            self.read_blank()
            self.read_comments()

    def parse(self):
        try:
            vertices = self.read_vertices()
            for src, dst in self.read_arcs(vertices):
                yield vertices[src], vertices[dst]
        except EOFError:
            # TODO(olegs): Make sure that we parsed everything
            pass


class Parser:
    '''
    Example:

    ::

        import cleanX.source.Dir as Dir
        import cleanX.source.Glob as Glob

        Source1 = Dir("/foo/bar")
        Source2 = Glob("/foo/*.jpg")

        Out1, Out2, Out3 = Acquire[arg1="foo", arg2=42](Source1, Source2)
        Out4 = Or[arg1=True](Out1, Out2)
        Out5 = Crop(Out3, Out4)
        Save[path="/foo/bar"](Out5)
    '''

    name_chars = string.ascii_letters + string.digits

    def __init__(self, stream):
        self.stream = stream
        self.ctors = get_known_steps()
        self.overflow = None
        # TODO(olegs): Track position in the parser
        self.line = 0
        self.column = 0

    def peek(self):
        if self.overflow is None:
            self.overflow = self.stream.read(1)
            if not self.overflow:
                raise EOFError()
        return self.overflow

    def consume(self):
        self.overflow = None

    def read_blank(self):
        while True:
            c = self.peek()
            if c not in string.whitespace:
                break
            self.consume()

    def raise_syntax_error(self, c):
        err = SyntaxError('Unexpected {!r}'.format(c))
        err.lineno = self.line
        err.offset = self.column
        raise err

    def read(self, cs):
        for c in cs:
            d = self.peek()
            if c != d:
                self.raise_syntax_error(d)
            self.consume()

    def read_arrow(self):
        self.read_blank()
        self.read('-')
        self.read('>')

    def read_step(self):
        raw = []
        while True:
            c = self.peek()
            if c in ('\n', '#'):
                break
            raw.append(c)
            self.consume()
        l, g = locals(), globals()
        l = dict(l)
        l.update(self.ctors)
        return eval(''.join(raw), l, g)

    def read_label(self):
        raw = []
        while True:
            c = self.peek()
            if c == ':':
                self.consume()
                break
            raw.append(c)
            self.consume()
        if not raw:
            self.raise_syntax_error('empty label')
        return ''.join(raw)

    def read_comment(self):
        self.read_blank()
        if self.peek() != '#':
            return False
        self.consume()

        while self.peek() != '\n':
            self.consume()
        return True

    def read_comments(self):
        while self.read_comment():
            pass

    def read_vertices(self):
        self.read_blank()
        self.read_comments()
        self.read_blank()
        self.read('vertices:')
        self.read_blank()
        self.read_comments()

        result = {}

        while True:
            label = self.read_label()
            if label == 'arcs':
                break
            if label in result:
                self.raise_syntax_error(
                    'duplicate label: {}'.format(label),
                )
            result[label] = self.read_step()
            self.read_comments()

        return result

    def read_word(self):
        result = []

        while True:
            c = self.peek()
            if c in string.whitespace:
                if not result:
                    self.raise_syntax_error('mising word')
                break
            result.append(c)
            self.consume()
        return ''.join(result)

    def read_arcs(self, labels):
        self.read_blank()
        self.read_comments()

        while True:
            self.read_blank()
            a = self.read_word()
            if a not in labels:
                self.raise_syntax_error(a)
            self.read_blank()
            self.read_arrow()
            self.read_blank()
            b = self.read_word()
            if b not in labels:
                self.raise_syntax_error(b)
            yield a, b
            self.read_blank()
            self.read_comments()

    def parse(self):
        try:
            vertices = self.read_vertices()
            for src, dst in self.read_arcs(vertices):
                yield vertices[src], vertices[dst]
        except EOFError:
            # TODO(olegs): Make sure that we parsed everything
            pass
