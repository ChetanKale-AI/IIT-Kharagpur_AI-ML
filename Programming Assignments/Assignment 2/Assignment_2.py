""" 
    Do not import any more libraries here 
"""
import sys

def frac(n: int) -> int:
    """
    Calculate the factorial of a given number n.
    
    Args:
    n (int): The number for which the factorial is to be calculated.
    
    Returns:
    int: The factorial of the number n.
    
    ATTENTION: Do not change anything here.
    """
    if n == 0:
        return 0
    if n == 1:
        return 1
    else:
        return n * frac(n - 1)
    
def series(ip: int) -> float:
    """
    Calculate the series sum based on the given integer ip and return the result.
    
    Args:
    ip (int): The integer input for the series calculation.
    
    Returns:
    float: The calculated sum of the series.
    
    You are required to modify this function. Remove the 'pass' keyword and write your own code
    """
    
    if ip < 0:
        return 999.0

    sum_series = 0.0
    for s in range(1, ip + 1):
        sum_series += frac(s) / (s * s)

    return round(sum_series, 2)



## DO NOT MODIFY THIS PART, Please maintain this structure
if __name__ == "__main__":
    """
    The main entry point of the program.
    
    ATTENTION: Do not change anything here.
    """
    value = series(int(sys.argv[1]))
    print(value)
    """ End to call """
