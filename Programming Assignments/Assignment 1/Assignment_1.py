## Programming Assignment 0
## Practice question for automatic grading

## Do not use any additional libraries
import sys


## Problem: Write a function that takes a list of integers as
## and returns the sum of all the integers in the list.

def sum_of_list(lst):
    ## Remove the pass statement and write your code here
    ## CODEBASE:S01 Start
    sum = 0
    for i in lst:
        sum += i
    return sum


    ## CODEBASE:S01 End

 


## Skeleton code for testing
## DO NOT MODIFY, Please maintain this structure
if __name__ == '__main__':

    """
    ALERT: * * * No changes are allowed in this section  * * *
    """

    numbers_str = sys.argv[1:]  
    numbers = [int(num) for num in numbers_str]

    # print("List of numbers:", numbers)
    print(sum_of_list(numbers)) 
    
    """ End to call """
