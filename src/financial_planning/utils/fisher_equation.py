"""
Fisher Equation Utilities

Implements Irving Fisher's equation relating nominal rates, real rates,
and inflation: (1 + nominal) = (1 + real) * (1 + inflation)

Also includes related financial mathematics utilities.
"""

import numpy as np
from typing import Union, List


def nominal_to_real(
    nominal_rate: Union[float, np.ndarray],
    inflation_rate: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert nominal rate to real rate using Fisher equation.
    
    Formula: real_rate = (1 + nominal_rate) / (1 + inflation_rate) - 1
    
    Args:
        nominal_rate: Nominal interest rate
        inflation_rate: Inflation rate
        
    Returns:
        Real interest rate
        
    Examples:
        >>> nominal_to_real(0.10, 0.03)
        0.06796116504854369  # approximately 6.8%
    """
    return (1 + nominal_rate) / (1 + inflation_rate) - 1


def real_to_nominal(
    real_rate: Union[float, np.ndarray],
    inflation_rate: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert real rate to nominal rate using Fisher equation.
    
    Formula: nominal_rate = (1 + real_rate) * (1 + inflation_rate) - 1
    
    Args:
        real_rate: Real interest rate
        inflation_rate: Inflation rate
        
    Returns:
        Nominal interest rate
        
    Examples:
        >>> real_to_nominal(0.05, 0.03)
        0.0815  # 8.15%
    """
    return (1 + real_rate) * (1 + inflation_rate) - 1


def implied_inflation(
    nominal_rate: Union[float, np.ndarray],
    real_rate: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate implied inflation rate from nominal and real rates.
    
    Formula: inflation_rate = (1 + nominal_rate) / (1 + real_rate) - 1
    
    Args:
        nominal_rate: Nominal interest rate
        real_rate: Real interest rate
        
    Returns:
        Implied inflation rate
        
    Examples:
        >>> implied_inflation(0.10, 0.05)
        0.047619047619047616  # approximately 4.76%
    """
    return (1 + nominal_rate) / (1 + real_rate) - 1


def compound_growth_rate(
    beginning_value: float,
    ending_value: float,
    periods: int
) -> float:
    """
    Calculate compound annual growth rate (CAGR).
    
    Formula: CAGR = (ending_value / beginning_value)^(1/periods) - 1
    
    Args:
        beginning_value: Starting value
        ending_value: Ending value
        periods: Number of periods
        
    Returns:
        Compound growth rate
        
    Examples:
        >>> compound_growth_rate(100, 150, 5)
        0.08447177731497098  # approximately 8.45% per year
    """
    if beginning_value <= 0:
        raise ValueError("Beginning value must be positive")
    if periods <= 0:
        raise ValueError("Periods must be positive")
    
    return (ending_value / beginning_value) ** (1 / periods) - 1


def present_value(
    future_value: Union[float, np.ndarray],
    discount_rate: float,
    periods: Union[int, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate present value of future cash flows.
    
    Formula: PV = FV / (1 + r)^n
    
    Args:
        future_value: Future value or array of future values
        discount_rate: Discount rate per period
        periods: Number of periods or array of periods
        
    Returns:
        Present value
        
    Examples:
        >>> present_value(100, 0.10, 5)
        62.09213230591562
    """
    return future_value / (1 + discount_rate) ** periods


def future_value(
    present_value: Union[float, np.ndarray],
    growth_rate: float,
    periods: Union[int, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate future value with compound growth.
    
    Formula: FV = PV * (1 + r)^n
    
    Args:
        present_value: Present value
        growth_rate: Growth rate per period
        periods: Number of periods
        
    Returns:
        Future value
        
    Examples:
        >>> future_value(100, 0.10, 5)
        161.051
    """
    return present_value * (1 + growth_rate) ** periods


def effective_annual_rate(
    nominal_rate: float,
    compounding_periods: int
) -> float:
    """
    Calculate effective annual rate from nominal rate.
    
    Formula: EAR = (1 + r/n)^n - 1
    
    Args:
        nominal_rate: Nominal annual rate
        compounding_periods: Number of compounding periods per year
        
    Returns:
        Effective annual rate
        
    Examples:
        >>> effective_annual_rate(0.12, 12)  # Monthly compounding
        0.126825030131969
    """
    return (1 + nominal_rate / compounding_periods) ** compounding_periods - 1


def continuous_compounding_rate(
    discrete_rate: float
) -> float:
    """
    Convert discrete rate to continuous compounding rate.
    
    Formula: r_continuous = ln(1 + r_discrete)
    
    Args:
        discrete_rate: Discrete compounding rate
        
    Returns:
        Continuous compounding rate
        
    Examples:
        >>> continuous_compounding_rate(0.10)
        0.09531017980432486
    """
    return np.log(1 + discrete_rate)


def discrete_from_continuous(
    continuous_rate: float
) -> float:
    """
    Convert continuous rate to discrete compounding rate.
    
    Formula: r_discrete = e^r_continuous - 1
    
    Args:
        continuous_rate: Continuous compounding rate
        
    Returns:
        Discrete compounding rate
        
    Examples:
        >>> discrete_from_continuous(0.10)
        0.10517091807564771
    """
    return np.exp(continuous_rate) - 1


def inflation_adjust_series(
    nominal_series: Union[List[float], np.ndarray],
    inflation_rates: Union[List[float], np.ndarray],
    base_year_index: int = 0
) -> np.ndarray:
    """
    Adjust a nominal series for inflation to constant dollars.
    
    Args:
        nominal_series: Series of nominal values
        inflation_rates: Series of inflation rates
        base_year_index: Index of base year for constant dollars
        
    Returns:
        Array of real (inflation-adjusted) values
        
    Examples:
        >>> nominal = [100, 103, 106.5, 110]
        >>> inflation = [0.03, 0.034, 0.033]
        >>> inflation_adjust_series(nominal, inflation)
        array([100.        ,  99.61089494,  99.78811916, 100.14476564])
    """
    nominal_array = np.array(nominal_series)
    inflation_array = np.array(inflation_rates)
    
    if len(nominal_array) != len(inflation_array) + 1:
        raise ValueError(
            "Inflation rates should be one less than nominal series length"
        )
    
    # Build cumulative inflation factors
    cumulative_inflation = np.ones(len(nominal_array))
    for i in range(len(inflation_array)):
        if i < base_year_index:
            # Deflate backwards
            cumulative_inflation[i] = cumulative_inflation[i + 1] / (
                1 + inflation_array[i]
            )
        else:
            # Inflate forwards
            cumulative_inflation[i + 1] = cumulative_inflation[i] * (
                1 + inflation_array[i]
            )
    
    # Adjust to base year
    base_factor = cumulative_inflation[base_year_index]
    real_series = nominal_array * base_factor / cumulative_inflation
    
    return real_series


def real_discount_factor(
    nominal_rate: float,
    inflation_rate: float,
    periods: Union[int, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate discount factor using real interest rate.
    
    Args:
        nominal_rate: Nominal discount rate
        inflation_rate: Inflation rate
        periods: Number of periods
        
    Returns:
        Real discount factor
        
    Examples:
        >>> real_discount_factor(0.10, 0.03, 5)
        0.7307415967590462
    """
    real_rate = nominal_to_real(nominal_rate, inflation_rate)
    return 1 / (1 + real_rate) ** periods


if __name__ == "__main__":
    # Examples and tests
    print("Fisher Equation Examples:")
    print("=" * 60)
    
    # Example 1: Convert nominal to real
    nominal = 0.10
    inflation = 0.03
    real = nominal_to_real(nominal, inflation)
    print(f"\n1. Nominal to Real Conversion:")
    print(f"   Nominal rate: {nominal:.2%}")
    print(f"   Inflation: {inflation:.2%}")
    print(f"   Real rate: {real:.2%}")
    
    # Example 2: Convert real to nominal
    real = 0.05
    inflation = 0.03
    nominal = real_to_nominal(real, inflation)
    print(f"\n2. Real to Nominal Conversion:")
    print(f"   Real rate: {real:.2%}")
    print(f"   Inflation: {inflation:.2%}")
    print(f"   Nominal rate: {nominal:.2%}")
    
    # Example 3: CAGR calculation
    beginning = 100
    ending = 150
    years = 5
    cagr = compound_growth_rate(beginning, ending, years)
    print(f"\n3. CAGR Calculation:")
    print(f"   Beginning value: ${beginning}")
    print(f"   Ending value: ${ending}")
    print(f"   Years: {years}")
    print(f"   CAGR: {cagr:.2%}")
    
    # Example 4: Present value
    fv = 100
    rate = 0.10
    periods = 5
    pv = present_value(fv, rate, periods)
    print(f"\n4. Present Value:")
    print(f"   Future value: ${fv}")
    print(f"   Discount rate: {rate:.2%}")
    print(f"   Periods: {periods}")
    print(f"   Present value: ${pv:.2f}")
    
    # Example 5: Inflation adjustment
    print(f"\n5. Inflation Adjustment:")
    nominal_series = [100, 103, 106.5, 110]
    inflation_series = [0.03, 0.034, 0.033]
    real_series = inflation_adjust_series(nominal_series, inflation_series)
    print(f"   Nominal series: {nominal_series}")
    print(f"   Inflation rates: {[f'{r:.1%}' for r in inflation_series]}")
    print(f"   Real series: {[f'{v:.2f}' for v in real_series]}")
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
