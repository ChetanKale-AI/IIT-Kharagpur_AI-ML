# Statistical Analysis of Human Measurements

This project delves into fundamental statistical methods applied to a dataset of human body measurements, with a focus on understanding key mathematical concepts necessary for AI/ML, such as point estimation, confidence intervals, covariance matrices, eigenvalue analysis, and linear regression.

# Dataset Overview

The dataset (SM_data.csv) includes the following variables:

- Height (H): Height of the individual in centimeters.
- Palm Length (PL): Length of the individual's palm in centimeters.
- Arm Length (AL): Length of the individual's arm in centimeters.
- Foot Length (FL): Length of the individual's foot in centimeters.
- Male/Female (MF): Gender of the individual ('M' for Male, 'F' for Female).

# Key Analyses and Results

**1. Data Summary:**

- Mean of Height: 166.71 cm
- Median of Height: 167.57 cm
- Variance of Height: 124.39 cm^2
- Range of Height: 132 cm to 183 cm

**2. Normal Distribution Fitting:**

The palm length (PL) was fit to a normal distribution with:
- Mean: 17.50 cm
- Standard Deviation: 3.07 cm
- The histogram with the fitted normal distribution curve is plotted.

![image](https://github.com/user-attachments/assets/c89c8ded-9ffd-4208-bcb7-d1fdfb37d7ba)

**3. Point and Interval Estimation:**

- MLE of the Mean of Palm Length: 17.50 cm
- Unbiased Estimate of Variance for Palm Length: 9.45 cm^2
- MLE of Variance for Palm Length: 9.33 cm^2
- 95% Confidence Interval for the Mean of Palm Length: (16.79 cm, 18.21 cm)
- 95% Confidence Interval for the Variance of Palm Length: (7.01 cm^2, 13.46 cm^2)

**4. Covariance and Eigenvalue Analysis:**

- Covariance matrix for height, palm length, arm length, and foot length:
```bash
     H         PL          AL         FL
H   124.39    16.75     3.16       14.76
PL   16.75     9.45     8.55        9.74
AL    3.16     8.55    197.28       9.89
FL   14.76     9.74     9.89       13.04
```

- Eigenvalues of the Covariance Matrix:

  - [198.56, 128.52, 15.85, 1.22]

- Proportion of Variation Explained:

  - First eigenvalue explains 57.69% of the variation.  
  - Cumulative proportion explained: [57.69%, 95.04%, 99.65%, 100%]  

- Correlation Matrix:
```bash
     H         PL         AL         FL
H   1.000    0.489     0.020      0.366
PL  0.489    1.000     0.198      0.878
AL  0.020    0.198     1.000      0.195
FL  0.366    0.878     0.195      1.000
```

**5. Regression Analysis:**

Fitting a linear regression model between height (H) and palm length (PL):  
Overall Model:  

  - Intercept: 135.71  
  - Coefficient for PL: 1.77
  - R-squared: 0.24 (PL explains 24% of the variation in height)  

- Separate Models for Genders:

  - Male:
- Intercept: 156.10
- Coefficient for PL: 0.90

  - Female:
- Intercept: 129.92
- Coefficient for PL: 2.01

![image](https://github.com/user-attachments/assets/8fe1632a-b79b-4af0-84eb-08efbef6a226)


- The regression lines are plotted, showing differences in height predictions based on palm length for males and females.

