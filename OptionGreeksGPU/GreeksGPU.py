import numpy as np
from math import log, exp, sqrt
from numba import cuda
import math

@cuda.jit(device=True)
def norm_cdf_gpu(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# -----------------------------------------------------------

@cuda.jit(device=True)
def norm_pdf_gpu(x):
    return math.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)


# -----------------------------------------------------------

@cuda.jit
def impliedVolatility_gpu(underlyingPrice,strikePrice,interestRate,daysToExpiration,Price,optionType,output):
    i = cuda.grid(1)
    if i < underlyingPrice.size:
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

@cuda.jit
def getDelta_gpu(underlyingPrice,strikePrice,interestRate,daysToExpiration,IV,optionType,deltas,a_values,d1_values,d2_values):
    i = cuda.grid(1)
    if i < underlyingPrice.size:
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

@cuda.jit
def getDelta2_gpu(interestRate,daysToExpiration,d2_values, optionType, delta2s):
    i = cuda.grid(1)
    if i < d2_values.size:
        _b_ = exp(-(interestRate * daysToExpiration))

        if optionType[i] == 0:
            delta2s[i] = -norm_cdf_gpu(d2_values[i]) * _b_
        elif optionType[i] == 1:
            delta2s[i] = norm_cdf_gpu(-d2_values[i]) * _b_


# -----------------------------------------------------------

@cuda.jit
def getVega_gpu(underlyingPrice,daysToExpiration,d1_values,vegas):
    i = cuda.grid(1)
    if i < underlyingPrice.size:
        vegas[i] = underlyingPrice[i] * norm_pdf_gpu(d1_values[i]) * sqrt(daysToExpiration) / 100


# -----------------------------------------------------------

@cuda.jit
def getGamma_gpu(underlyingPrice,d1_values,a_values,gammas):
    i = cuda.grid(1)
    if i < underlyingPrice.size:
        gammas[i] = norm_pdf_gpu(d1_values[i]) / (underlyingPrice[i] * a_values[i])


# -----------------------------------------------------------

@cuda.jit
def getTheta_gpu(underlyingPrice,strikePrice,interestRate,daysToExpiration,IV,d1_values,d2_values,optionType,thetas):
    i = cuda.grid(1)
    if i < underlyingPrice.size:
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

@cuda.jit
def getRho_gpu(strikePrice,interestRate,daysToExpiration,d2_values,optionType,rhos):
    i = cuda.grid(1)
    if i < strikePrice.size:
        _b_ = exp(-(interestRate * daysToExpiration))
        if optionType[i] == 0:
            rhos[i] = strikePrice[i] * daysToExpiration * _b_ * norm_cdf_gpu(d2_values[i]) / 100
        elif optionType[i] == 1:
            rhos[i] = -strikePrice[i] * daysToExpiration * _b_ * norm_cdf_gpu(-d2_values[i]) / 100


# -----------------------------------------------------------

@cuda.jit
def getParity_gpu(underlyingPrices,strikePrices,interestRate,daysToExpiration,callPrices,putPrices,output):
    pos = cuda.grid(1)

    if pos < underlyingPrices.size:
        pv_strike = strikePrices[pos] / ((1 + interestRate) ** daysToExpiration)
        output[pos] = callPrices[pos] - putPrices[pos] - underlyingPrices[pos] + pv_strike


# -----------------------------------------------------------
# -----------------------------------------------------------

def calculate_option_metrics(option_data, days_to_expiry, interest_rate):
    interest_rate = interest_rate / 100
    days_to_expiry = days_to_expiry / 365

    strikePrices = cuda.to_device(option_data[:, 0].astype(np.float32))
    underlyingPrices = cuda.to_device(option_data[:, 1].astype(np.float32))

    callPrices = cuda.to_device(option_data[:, 2].astype(np.float32))
    call_optionTypes = cuda.to_device(option_data[:, 3].astype(np.int8))

    putPrices = cuda.to_device(option_data[:, 4].astype(np.float32))
    put_optionTypes = cuda.to_device(option_data[:, 5].astype(np.int8))

    call_IVs = cuda.to_device(np.zeros_like(underlyingPrices))
    call_deltas = cuda.to_device(np.zeros_like(underlyingPrices))
    call_a_values = cuda.to_device(np.zeros_like(underlyingPrices))
    call_d1_values = cuda.to_device(np.zeros_like(underlyingPrices))
    call_d2_values = cuda.to_device(np.zeros_like(underlyingPrices))
    call_delta2s = cuda.to_device(np.zeros_like(underlyingPrices))
    call_vegas = cuda.to_device(np.zeros_like(underlyingPrices))
    call_gammas = cuda.to_device(np.zeros_like(underlyingPrices))
    call_thetas = cuda.to_device(np.zeros_like(underlyingPrices))
    call_rhos = cuda.to_device(np.zeros_like(underlyingPrices))

    put_IVs = cuda.to_device(np.zeros_like(underlyingPrices))
    put_deltas = cuda.to_device(np.zeros_like(underlyingPrices))
    put_a_values = cuda.to_device(np.zeros_like(underlyingPrices))
    put_d1_values = cuda.to_device(np.zeros_like(underlyingPrices))
    put_d2_values = cuda.to_device(np.zeros_like(underlyingPrices))
    put_delta2s = cuda.to_device(np.zeros_like(underlyingPrices))
    put_vegas = cuda.to_device(np.zeros_like(underlyingPrices))
    put_gammas = cuda.to_device(np.zeros_like(underlyingPrices))
    put_thetas = cuda.to_device(np.zeros_like(underlyingPrices))
    put_rhos = cuda.to_device(np.zeros_like(underlyingPrices))

    parity = cuda.to_device(np.zeros_like(underlyingPrices))

    threadsperblock = 14
    blockspergrid = (underlyingPrices.size + (threadsperblock - 1)) // threadsperblock
    impliedVolatility_gpu[blockspergrid, threadsperblock](underlyingPrices,
                                                          strikePrices,
                                                          interest_rate,
                                                          days_to_expiry,
                                                          callPrices,
                                                          call_optionTypes,
                                                          call_IVs)
    impliedVolatility_gpu[blockspergrid, threadsperblock](underlyingPrices,
                                                          strikePrices,
                                                          interest_rate,
                                                          days_to_expiry,
                                                          putPrices,
                                                          put_optionTypes,
                                                          put_IVs)

    getDelta_gpu[blockspergrid, threadsperblock](underlyingPrices,
                                                 strikePrices,
                                                 interest_rate,
                                                 days_to_expiry,
                                                 call_IVs,
                                                 call_optionTypes,
                                                 call_deltas,
                                                 call_a_values,
                                                 call_d1_values,
                                                 call_d2_values)
    getDelta_gpu[blockspergrid, threadsperblock](underlyingPrices,
                                                 strikePrices,
                                                 interest_rate,
                                                 days_to_expiry,
                                                 put_IVs,
                                                 put_optionTypes,
                                                 put_deltas,
                                                 put_a_values,
                                                 put_d1_values,
                                                 put_d2_values)

    getDelta2_gpu[blockspergrid, threadsperblock](interest_rate,
                                                  days_to_expiry,
                                                  call_d2_values,
                                                  call_optionTypes,
                                                  call_delta2s)
    getDelta2_gpu[blockspergrid, threadsperblock](interest_rate,
                                                  days_to_expiry,
                                                  put_d2_values,
                                                  put_optionTypes,
                                                  put_delta2s)

    getVega_gpu[blockspergrid, threadsperblock](underlyingPrices,
                                                days_to_expiry,
                                                call_d1_values,
                                                call_vegas)
    getVega_gpu[blockspergrid, threadsperblock](underlyingPrices,
                                                days_to_expiry,
                                                put_d1_values,
                                                put_vegas)

    getGamma_gpu[blockspergrid, threadsperblock](underlyingPrices,
                                                 call_d1_values,
                                                 call_a_values,
                                                 call_gammas)
    getGamma_gpu[blockspergrid, threadsperblock](underlyingPrices,
                                                 put_d1_values,
                                                 put_a_values,
                                                 put_gammas)

    getTheta_gpu[blockspergrid, threadsperblock](underlyingPrices,
                                                 strikePrices,
                                                 interest_rate,
                                                 days_to_expiry,
                                                 call_IVs,
                                                 call_d1_values,
                                                 call_d2_values,
                                                 call_optionTypes,
                                                 call_thetas)
    getTheta_gpu[blockspergrid, threadsperblock](underlyingPrices,
                                                 strikePrices,
                                                 interest_rate,
                                                 days_to_expiry,
                                                 put_IVs,
                                                 put_d1_values,
                                                 put_d2_values,
                                                 put_optionTypes,
                                                 put_thetas)

    getRho_gpu[blockspergrid, threadsperblock](strikePrices,
                                               interest_rate,
                                               days_to_expiry,
                                               call_d2_values,
                                               call_optionTypes,
                                               call_rhos)
    getRho_gpu[blockspergrid, threadsperblock](strikePrices,
                                               interest_rate,
                                               days_to_expiry,
                                               put_d2_values,
                                               put_optionTypes,
                                               put_rhos)

    return [call_IVs.copy_to_host(),
            call_deltas.copy_to_host(),
            call_delta2s.copy_to_host(),
            call_vegas.copy_to_host(),
            call_gammas.copy_to_host(),
            call_thetas.copy_to_host(),
            call_rhos.copy_to_host(),
            put_IVs.copy_to_host(),
            put_deltas.copy_to_host(),
            put_delta2s.copy_to_host(),
            put_vegas.copy_to_host(),
            put_gammas.copy_to_host(),
            put_thetas.copy_to_host(),
            put_rhos.copy_to_host()]
