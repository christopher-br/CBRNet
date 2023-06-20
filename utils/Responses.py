import numpy as np
import scipy

def generalized_logistic(d, i1, i2, i3, i4, scaling_parameter=1):
    """
    Function to generate point on a generalized logistic function per a defined dose d.
    The fct is parameterized by 4 inputs.
    
    Parameters:
        d (float): The dosage element (0,1)
        i1 to i4 (float): The inputs element (0,1)
    """
    A = float(0.2*(i1)) # The left asymptote
    K = float(0.7 + 0.1*(i2)) # The right asymptote
    B = float(0.5 + 5*i3) # The growth rate of steepness
    NU = float(1) # Where the maximum growth occurs
    Q = float(1) # Relates to Y(0)
    C = float(1) # Affects right asymptote
    P = float(i4) # Turning point of the curve
    
    y = float(scaling_parameter) * (A + ((K - A)/((C + Q * np.exp(-B*(10*(d-P))))**(1/NU))))
    return y


def stacked_sigmoid(d, i1, i2, i3, i4, scaling_parameter=1):
    """
    Function to generate point on a stacked sigmoid function per a defined dose d.
    The fct is parameterized by 4 inputs.
    
    Parameters:
        d (float): The dosage element (0,1)
        i1 to i4 (float): The inputs element (0,1)
    """
    
    y = float(scaling_parameter) * ((0.2 * i1) +
         ((0.8 * i2) * logistic((d / i3))) +
         ((1 - 0.2 * i1 - 0.8 * i2) * i3 * logistic(((d - i4) / (1 - i4))))
         )
    return y

def logistic(dosage):
    """
    Function to generate point on a sigmoid function per a defined dose d.
    
    Parameters:
        d (float): The dosage element (0,1)
    """
    y = (0 + ((1 - 0)/((1 + 1 * np.exp(-2*(10*(dosage-0.5))))**(1/1))))
    return y

def quadratic(d, i1, i2, i3, i4, scaling_parameter=1):
    """
    Function to generate point on a quadratic function per a defined dose d.
    The fct is parameterized by 4 inputs.
    
    Parameters:
        d (float): The dosage element (0,1)
        i1 to i4 (float): The inputs element (0,1)
    """
    
    y = float(scaling_parameter) * (i1 + 12.0 * d * (d - 0.75 * (
                i2 / i3)) ** 2)
    return y

def sinewave(d, i1, i2, i3, i4, scaling_parameter=1):
    """
    Function to generate point on a sinewave-type function per a defined dose d.
    The fct is parameterized by 4 inputs.
    
    Parameters:
        d (float): The dosage element (0,1)
        i1 to i4 (float): The inputs element (0,1)
    """
    
    y = float(scaling_parameter) * (i1 + np.sin(
            np.pi * (i2 / i3) * d))
    return y

def linear(d, i1, i2, i3, i4, scaling_parameter=1):
    """
    Function to generate point on a polynomial-type function per a defined dose d.
    The fct is parameterized by 4 inputs.
    
    Parameters:
        d (float): The dosage element (0,1)
        i1 to i4 (float): The inputs element (0,1)
    """
    
    y = float(scaling_parameter) * (i1 + 12.0 * (i2 * d - i3 * d ** 2))
    return y

def get_outcome(x, v, d, response_type, scaling_parameter=10):
    """
    Gets outcome of an observation according to specified dose response, parameterization, and assigned dose
    
    Parameters:
        x (np array): The observation
        v (np.array): The linear coefficients to calculate dose response parameters
        d (float): The dose. Element (0,1)
        response_type (str): Which response to use
        scaling_parameter (float): A scaling parameter with which to scale the outcome
    """
    # Calculate inputs
    input1 = np.dot(x, v[0])
    input2 = np.dot(x, v[1])
    input3 = np.dot(x, v[2])
    input4 = np.dot(x, v[3])
    
    # Define response fct dict
    response_map = {'richards': generalized_logistic,
                    'stacked_sigmoid': stacked_sigmoid,
                    'quadratic': quadratic,
                    'sine': sinewave,
                    'linear': linear
                    }
    
    # Calculate the response
    y = response_map[response_type](d, input1, input2, input3, input4, scaling_parameter)
    return y
