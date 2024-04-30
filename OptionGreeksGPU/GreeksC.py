import numpy as np
from math import log, exp, sqrt
import math
from numba import jit

@jit(nopython=True)
def norm_cdf_gpu(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# -----------------------------------------------------------

@jit(nopython=True)
def norm_pdf_gpu(x):
    return math.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)


# -----------------------------------------------------------

@jit(nopython=True)
def impliedVolatility_C(underlyingPrice,strikePrice,interestRate,daysToExpiration,Price,optionType,output):
    for i in range(len(underlyingPrice)):
        high = 500.0
        low = 0.0
        mid = 0.0
        for _ in range(10000):

            mid = (high + low) / 2

            if mid < 0.00001:
                mid = 0.00001

            _a_ = mid / 100 * math.sqrt(daysToExpiration)
            _d1_ = (math.log(underlyingPrice[i] / strikePrice[i]) + (
                    interestRate + (mid / 100 ** 2) / 2) * daysToExpiration) / _a_
            _d2_ = _d1_ - _a_

            estimate = 0.0

            if daysToExpiration == 0:
                if optionType[i] == 0:
                    estimate = max(0.0, underlyingPrice[i] - strikePrice[i])
                elif optionType[i] == 1:
                    estimate = max(0.0, strikePrice[i] - underlyingPrice[i])
            else:
                if optionType[i] == 0:
                    estimate = underlyingPrice[i] * norm_cdf_gpu(_d1_) - strikePrice[i] * exp(
                        -interestRate * daysToExpiration) * norm_cdf_gpu(_d2_)
                elif optionType[i] == 1:
                    estimate = strikePrice[i] * exp(-interestRate * daysToExpiration) * norm_cdf_gpu(-_d2_) - \
                               underlyingPrice[i] * norm_cdf_gpu(-_d1_)

            if round(estimate, 2) == Price[i]:
                break
            elif estimate > Price[i]:
                high = mid
            else:
                low = mid

        output[i] = mid


# -----------------------------------------------------------

@jit(nopython=True)
def getDelta_C(underlyingPrice,strikePrice,interestRate,daysToExpiration,IV,optionType,deltas,a_values,d1_values,d2_values):
    for i in range(len(underlyingPrice)):
        _a_ = IV[i] / 100 * sqrt(daysToExpiration)
        _d1_ = (log(underlyingPrice[i] / strikePrice[i]) + (
                interestRate + (IV[i] / 100 ** 2) / 2) * daysToExpiration) / _a_
        _d2_ = _d1_ - _a_

        if optionType[i] == 0:
            deltas[i] = norm_cdf_gpu(_d1_)
        elif optionType[i] == 1:
            deltas[i] = -norm_cdf_gpu(-_d1_)

        a_values[i] = _a_
        d1_values[i] = _d1_
        d2_values[i] = _d2_


# -----------------------------------------------------------

@jit(nopython=True)
def getDelta2_C(interestRate,daysToExpiration,d2_values, optionType, delta2s):
    for i in range(len(d2_values)):
        _b_ = exp(-(interestRate * daysToExpiration))

        if optionType[i] == 0:
            delta2s[i] = -norm_cdf_gpu(d2_values[i]) * _b_
        elif optionType[i] == 1:
            delta2s[i] = norm_cdf_gpu(-d2_values[i]) * _b_


# -----------------------------------------------------------

@jit(nopython=True)
def getVega_C(underlyingPrice,daysToExpiration,d1_values,vegas):
    for i in range(len(underlyingPrice)):
        vegas[i] = underlyingPrice[i] * norm_pdf_gpu(d1_values[i]) * sqrt(daysToExpiration) / 100


# -----------------------------------------------------------

@jit(nopython=True)
def getGamma_C(underlyingPrice,d1_values,a_values,gammas):
    for i in range(len(underlyingPrice)):
        gammas[i] = norm_pdf_gpu(d1_values[i]) / (underlyingPrice[i] * a_values[i])


# -----------------------------------------------------------

@jit(nopython=True)
def getTheta_C(underlyingPrice,strikePrice,interestRate,daysToExpiration,IV,d1_values,d2_values,optionType,thetas):
    for i in range(len(underlyingPrice)):
        _b_ = exp(-(interestRate * daysToExpiration))
        if optionType[i] == 0:
            thetas[i] = (-underlyingPrice[i] * norm_pdf_gpu(d1_values[i]) * IV[i] / 100 / (
                        2 * sqrt(daysToExpiration)) - interestRate * strikePrice[i] * _b_ * norm_cdf_gpu(
                d2_values[i])) / 365
        elif optionType[i] == 1:
            thetas[i] = (-underlyingPrice[i] * norm_pdf_gpu(d1_values[i]) * IV[i] / 100 / (
                        2 * sqrt(daysToExpiration)) + interestRate * strikePrice[i] * _b_ * norm_cdf_gpu(
                -d2_values[i])) / 365


# -----------------------------------------------------------

@jit(nopython=True)
def getRho_C(strikePrice,interestRate,daysToExpiration,d2_values,optionType,rhos):
    for i in range(len(strikePrice)):
        _b_ = exp(-(interestRate * daysToExpiration))
        if optionType[i] == 0:
            rhos[i] = strikePrice[i] * daysToExpiration * _b_ * norm_cdf_gpu(d2_values[i]) / 100
        elif optionType[i] == 1:
            rhos[i] = -strikePrice[i] * daysToExpiration * _b_ * norm_cdf_gpu(-d2_values[i]) / 100


# -----------------------------------------------------------

@jit(nopython=True)
def getParity_C(underlyingPrices,strikePrices,interestRate,daysToExpiration,callPrices,putPrices,output):
    for i in range(len(underlyingPrices)):
        pv_strike = strikePrices[pos] / ((1 + interestRate) ** daysToExpiration)
        output[pos] = callPrices[pos] - putPrices[pos] - underlyingPrices[pos] + pv_strike


# -----------------------------------------------------------
# -----------------------------------------------------------

def calculate_option_metrics(option_data, days_to_expiry, interest_rate):
    interest_rate = interest_rate / 100
    days_to_expiry = days_to_expiry / 365

    strikePrices = np.array(option_data[:, 0].astype(np.float32))
    underlyingPrices = np.array(option_data[:, 1].astype(np.float32))

    callPrices = np.array(option_data[:, 2].astype(np.float32))
    call_optionTypes = np.array(option_data[:, 3].astype(np.int8))

    putPrices = np.array(option_data[:, 4].astype(np.float32))
    put_optionTypes = np.array(option_data[:, 5].astype(np.int8))

    call_IVs = np.zeros_like(underlyingPrices)
    call_deltas = np.zeros_like(underlyingPrices)
    call_a_values = np.zeros_like(underlyingPrices)
    call_d1_values = np.zeros_like(underlyingPrices)
    call_d2_values = np.zeros_like(underlyingPrices)
    call_delta2s = np.zeros_like(underlyingPrices)
    call_vegas = np.zeros_like(underlyingPrices)
    call_gammas = np.zeros_like(underlyingPrices)
    call_thetas = np.zeros_like(underlyingPrices)
    call_rhos = np.zeros_like(underlyingPrices)

    put_IVs = np.zeros_like(underlyingPrices)
    put_deltas = np.zeros_like(underlyingPrices)
    put_a_values = np.zeros_like(underlyingPrices)
    put_d1_values = np.zeros_like(underlyingPrices)
    put_d2_values = np.zeros_like(underlyingPrices)
    put_delta2s = np.zeros_like(underlyingPrices)
    put_vegas = np.zeros_like(underlyingPrices)
    put_gammas = np.zeros_like(underlyingPrices)
    put_thetas = np.zeros_like(underlyingPrices)
    put_rhos = np.zeros_like(underlyingPrices)

    parity = np.zeros_like(underlyingPrices)

    impliedVolatility_C(underlyingPrices,
                          strikePrices,
                          interest_rate,
                          days_to_expiry,
                          callPrices,
                          call_optionTypes,
                          call_IVs)

    impliedVolatility_C(underlyingPrices,
                          strikePrices,
                          interest_rate,
                          days_to_expiry,
                          putPrices,
                          put_optionTypes,
                          put_IVs)

    getDelta_C(underlyingPrices,
                 strikePrices,
                 interest_rate,
                 days_to_expiry,
                 call_IVs,
                 call_optionTypes,
                 call_deltas,
                 call_a_values,
                 call_d1_values,
                 call_d2_values)

    getDelta_C(underlyingPrices,
                 strikePrices,
                 interest_rate,
                 days_to_expiry,
                 put_IVs,
                 put_optionTypes,
                 put_deltas,
                 put_a_values,
                 put_d1_values,
                 put_d2_values)

    getDelta2_C(interest_rate,
                  days_to_expiry,
                  call_d2_values,
                  call_optionTypes,
                  call_delta2s)

    getDelta2_C(interest_rate,
                  days_to_expiry,
                  put_d2_values,
                  put_optionTypes,
                  put_delta2s)

    getVega_C(underlyingPrices,
                days_to_expiry,
                call_d1_values,
                call_vegas)

    getVega_C(underlyingPrices,
                days_to_expiry,
                put_d1_values,
                put_vegas)

    getGamma_C(underlyingPrices,
                 call_d1_values,
                 call_a_values,
                 call_gammas)

    getGamma_C(underlyingPrices,
                 put_d1_values,
                 put_a_values,
                 put_gammas)

    getTheta_C(underlyingPrices,
                 strikePrices,
                 interest_rate,
                 days_to_expiry,
                 call_IVs,
                 call_d1_values,
                 call_d2_values,
                 call_optionTypes,
                 call_thetas)

    getTheta_C(underlyingPrices,
                 strikePrices,
                 interest_rate,
                 days_to_expiry,
                 put_IVs,
                 put_d1_values,
                 put_d2_values,
                 put_optionTypes,
                 put_thetas)

    getRho_C(strikePrices,
               interest_rate,
               days_to_expiry,
               call_d2_values,
               call_optionTypes,
               call_rhos)

    getRho_C(strikePrices,
               interest_rate,
               days_to_expiry,
               put_d2_values,
               put_optionTypes,
               put_rhos)

    return [call_IVs,
            call_deltas,
            call_delta2s,
            call_vegas,
            call_gammas,
            call_thetas,
            call_rhos,
            put_IVs,
            put_deltas,
            put_delta2s,
            put_vegas,
            put_gammas,
            put_thetas,
            put_rhos]
