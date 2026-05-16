class ParameterUtils:
    @staticmethod
    def str_shape_to_tuple(target_shape: str):
        parsed_shape: tuple = ()
        if type(target_shape) is str:
            parsed_shape = tuple([int(dim_size) for dim_size in target_shape.split(",")])
        
        return parsed_shape
