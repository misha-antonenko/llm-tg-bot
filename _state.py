from copy import copy

import pydantic


class State(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __deepcopy__(self, memo):
        return copy(self)
