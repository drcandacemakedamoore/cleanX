# -*- coding: utf-8 -*-

import ast
import importlib

from collections import namedtuple, Counter

from lark import Lark, Transformer, v_args


class MissingDefinition(LookupError):

    def __init__(self, missing):
        super().__init__('Missing definition for {}'.format(missing))


class DuplicateVariables(ValueError):

    def __init__(self, duplicates):
        super().__init__('Found duplicate variables: {}'.format(duplicates))


class DuplicateOptions(ValueError):

    def __init__(self, duplicates):
        super().__init__('Found duplicate options: {}'.format(duplicates))


StepCall = namedtuple(
    'StepCall',
    'definition,options,variables,serial,splitter,joiner',
)
PipelineDef = namedtuple('PipelineDef', 'steps,goal')


cleanx_grammar = """
    ?start: pipeline -> create_pipeline
    ?pipeline: "pipeline" "(" definitions steps goal ")" -> pipeline
    ?definitions: "definitions" "(" dassign+ ")" -> definitions
    ?steps: "steps" "(" sassign+ ")" -> steps
    ?goal: "goal" "(" step ")" -> goal
    ?dassign: var "=" src -> dassign
    ?var: NAME
    ?src: module ":" def -> src
    ?module: NAME ("." NAME)*
    ?def: NAME
    ?sassign: var+ "=" step -> sassign
    ?step: var options? "(" var* ")" joiner? splitter? -> step
    ?options: "[" oassign+ "]" -> options
    ?oassign: var "=" val -> oassign
    ?joiner: "*" applier
    ?splitter: "/" applier
    ?applier: def "(" options? ")"
    ?val: number
        | string
        | boolean
        | null
        | list
    ?number: NUMBER -> number
    ?string: STRING -> string
    ?null: "null" -> null
    ?boolean: /true|false/ -> boolean
    ?list: "(" val* ")"
    COMMENT: /#.*$/

    %import common.ESCAPED_STRING -> STRING
    %import common.SIGNED_NUMBER -> NUMBER
    %import common.CNAME -> NAME
    %import common.WS
    %ignore COMMENT
    %ignore WS
"""


@v_args(inline=True)
class Parser(Transformer):
    '''
    Example:

    ::

        pipeline(
            definitions(
                dir = cleanX.source:Dir
                glob = cleanX.source:Glob
                acquire = cleanX.steps:Acquire
                or = cleanX.steps:Or
                crop = cleanX.steps:Crop
            )
            steps(
                source1 = dir[path = "/foo/bar"]()
                source2 = glob[pattern = "/foo/*.jpg"]()

                # Two sources will be averaged producing single source
                # and then duplicated into three different variables.
                # I.e. out1, out2 and out3 should be the same files.
                out1 out2 out3 = acquire[arg1 = "foo" arg2 = 42](
                    source1 source2
                ) * avg(width="min" height="max") / tee()
                out4 = or[arg1 = true](out1 out2)
                out5 = crop(out3 out4)
            )
            goal(
                save[path = "/foo/bar"](out5)
            )
        )
    '''

    def __init__(self):
        self.lark = Lark(
            cleanx_grammar,
            parser='lalr',
            transformer=self,
        )
        self._definitions = {}
        self._variables = {}

    def parse(self, stream):
        return self.lark.parse(stream).children[0]

    def number(self, n):
        return ast.literal_eval(n)

    def string(self, s):
        return ast.literal_eval(s)

    def null(self):
        return None

    def boolean(self, b):
        return b == 'true'

    def src(self, module, definition):
        mod = importlib.import_module('.'.join(module.children))
        return getattr(mod, definition)

    def dassign(self, name, definition):
        self._definitions[str(name)] = definition
        return definition

    def oassign(self, name, value):
        return str(name), value

    def options(self, *raw):
        varnames = tuple(n for n, _ in raw)
        duplicates = []
        for var, count in Counter(varnames).most_common():
            if count == 1:
                break
            duplicates.append(var)
        if duplicates:
            raise DuplicateOptions(duplicates)
        return raw

    def step(self, *args):
        try:
            definition = self._definitions[args[0]]
        except KeyError:
            raise MissingDefinition(args[0])
        options, variables = (), ()
        if len(args) > 1:
            if type(args[1]) is tuple:
                options = args[1]
                variables = tuple(str(a) for a in args[2:])
            else:
                variables = tuple(str(a) for a in args[1:])
        # TODO(wvxvw): Parse these
        serial = True
        splitter = None
        joiner = None
        # TODO(wvxvw): Here we could validate optional arguments to
        # steps.
        return StepCall(
            definition,
            options,
            variables,
            serial,
            splitter,
            joiner,
        )

    def sassign(self, *args):
        variables = tuple(str(s) for s in args[:-1])
        duplicates = []
        for var, count in Counter(variables).most_common():
            if count == 1:
                break
            duplicates.append(var)

        for v in variables:
            if v in self._variables:
                duplicates.append(v)
        if duplicates:
            raise DuplicateVariables(duplicates)
        for v in variables:
            self._variables[v] = args[-1]
        return args[-1]

    def pipeline(self, *args):
        # TODO(wvxvw): Here we could check for unused definitions, but
        # it's not very important for now.
        return PipelineDef(self._variables, args[-1].children[0])
