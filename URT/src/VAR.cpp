//=================================================================================================
//                    Copyright (C) 2016 Olivier Mallet - All Rights Reserved                      
//=================================================================================================

#include "../include/URT.hpp"

namespace urt {

//=================================================================================================

// parameter constructor for computing VAR test for a given number of lags
template <typename T>
VAR<T>::VAR(const Vector<T>& data, int lags, const std::string& trend, bool regression) : ur(data, lags, trend, regression)
{
   ur::test_name = _test_name;
   ur::valid_trends = _valid_trends;
}

//*************************************************************************************************

// parameter constructor for computing VAR test with lag length optimization
template <typename T>
VAR<T>::VAR(const Vector<T>& data, const std::string& method, const std::string& trend, bool regression) : ur(data, method, trend, regression)
{
   ur::test_name = _test_name;
   ur::valid_trends = _valid_trends;
}

//*************************************************************************************************

// compute test statistic
template <typename T>
const T& VAR<T>::statistic()
{
   // setting type of lags (if a default type of lags value has been chosen)
   ur::set_lags_type();
   // setting optimization method
   ur::set_method();
   // setting number of lags
   ur::set_lags();
   // setting regression trend
   ur::set_trend();
   // computing VAR test
   ur::compute_VAR();

   return ur::stat;
}

//*************************************************************************************************

// compute test p-value
template <class T>
const T& VAR<T>::pvalue()
{
   // computing test statistic
   this->statistic();
   // setting critical values coefficients pointer
   ur::coeff_ptr = &coeff_VAR.at(ur::trend);
   // computing p-value
   ur::pvalue();

   return ur::pval;
} 

//*************************************************************************************************

// output test results
template <class T>
void VAR<T>::show()
{
   // in case user modified test type, for VAR it should always be empty
   ur::test_type = std::string();  
   // computing p-value
   this->pvalue();
   // outputting results
   ur::show();
}

//=================================================================================================

}

