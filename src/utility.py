# ===================================================================
# Utility functions for different time marching schemes
# ===================================================================

def Euler(f, L, eps, dt):
    return f + dt*L(f, eps)

def RK2(f, L, eps, dt):
    k1 = L(f, eps)
    return f + dt*L(f + 0.5*dt*k1, eps)

def RK3(f, L, eps, dt):
    k1 = L(f, eps)
    k2 = L(f + 0.5*dt*k1, eps)
    k3 = L(f - dt*k1 + 2*dt*k2, eps)
    return f + 1/6*dt*(k1 + 4*k2 + k3)

def RK4(f, L, eps, dt):
    k1 = L(f, eps)
    k2 = L(f + 0.5*dt*k1, eps)
    k3 = L(f + 0.5*dt*k2, eps)
    k4 = L(f + dt*k3, eps)
    return f + dt*(k1 + 2*k2 + 2*k3 + k4)/6

# ===================================================================
# Utility functions for plotting 2d function
# ===================================================================

# Plotting 2d function f
def plot_2d(f, cs):
    N = f.shape[0]
    dv = 2*L/N
    v = np.mgrid[-L+dv/2:L+dv/2:dv,-L+dv/2:L+dv/2:dv]
    v_norm = v[0]**2 + v[1]**2

    fig, ax = plt.subplots()
    cs = ax.contour(v[0], v[1], f)
    if cs == 1:
        ax.clabel(cs, inline=0.5)
        
    ax.grid(linestyle=':')

    plt.show()


# ===================================================================
# Utility functions for reading config files
# ===================================================================
import json

class DictionaryUtility:
    """
    Utility methods for dealing with dictionaries.
    """
    @staticmethod
    def to_object(item):
        """
        Convert a dictionary to an object (recursive).
        """
        def convert(item):
            if isinstance(item, dict):
                return type('jo', (), {k: convert(v) for k, v in item.items()})
            if isinstance(item, list):
                def yield_convert(item):
                    for index, value in enumerate(item):
                        yield convert(value)
                return list(yield_convert(item))
            else:
                return item

        return convert(item)

    def to_dict(obj):
        """
         Convert an object to a dictionary (recursive).
         """
        def convert(obj):
            if not hasattr(obj, "__dict__"):
                return obj
            result = {}
            for key, val in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                element = []
                if isinstance(val, list):
                    for item in val:
                        element.append(convert(item))
                else:
                    element = convert(val)
                result[key] = element
            return result

        return convert(obj)


def get_config(config_path):
    with open(config_path) as json_data_file:
        config = json.load(json_data_file)
    # convert dict to object recursively for easy call
    config = DictionaryUtility.to_object(config)
    return config