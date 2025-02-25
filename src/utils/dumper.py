from functools import partial
import yaml

data = yaml.safe_load('''
swagger: '2.0'
host: api.petstore.com
basePath: /api/v1
schemas:
- https
consumes:
- application/json
''')


class MyDumper(yaml.SafeDumper):
    """
    Insert blank lines between top-level objects.
    
    Source: https://github.com/yaml/pyyaml/issues/127#issuecomment-525800484
    """
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()

dump_config = partial(
    yaml.dump, 
    Dumper=MyDumper, 
    sort_keys=False
)
