// DESCRIPTION: Appropriate license header at the top, e.g.,
/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// DESCRIPTION: A brief description of the overall purpose and contents of this
// file. Note the use of the @file for file descriptions DESCRIPTION: The text
// on the same line as the @brief will show up in lists and summaries, whereas
// the detailed text following on the next line is the detailed description.
/**---------------------------------------------------------------------------*
 * @file example_documentation.cpp
 * @brief Example code documentation for libgdf.
 *
 * This file provides examples of how source files, classes, functions, and
 * variables should be documented in libgdf.
 *---------------------------------------------------------------------------**/

// DESCRIPTION: A brief description of the purpose and functionality of the
// class
/**---------------------------------------------------------------------------*
 * @brief  This class serves as an example of how classes in libgdf should
 * be documented.
 *
 * In detail, this class shows how member functions and member variables should
 * be documented.
 *
 * @tparam T Template parameter for this class is used for...
 *---------------------------------------------------------------------------**/
template <typename T>
class example_class {
  // DESCRIPTION: Trivial class functions should be given names that make their
  // purpose clear DESCRIPTION: If their name makes the functionality obvious,
  // no documentation is necessary
  void get_my_int() { return my_int; }
  void set_my_int(int new_value) { my_int = new_value; }

  // DESCRIPTION: Non-trivial member functions should have a brief description
  // of the function as well as all of its parameters. Every parameter should be
  // decorated to indicate if it is an input or output parameter, or both with
  // @Param[in], @Param[out], and @Param[in,out] respectively.
  /**---------------------------------------------------------------------------*
   * @brief This is a complicated function that requires more detailed
   * documentation.
   *
   * Here is the more detailed description of what this function does and what
   * its logic is.
   *
   * @param[in] first_parameter  This parameter is an input parameter to the
   * function
   * @param[in,out] second_parameter This parameter is used both as an input and
   * output
   * @param[out] third_parameter This parameter is an output of the function
   *
   * @return The result of the complex function
   *---------------------------------------------------------------------------**/
  T complicated_function(int const first_parameter, double* second_parameter,
                         float* third_parameter) {
    // DESCRIPTION: Notice the use of *human readable* variable names. Human
    // readable variable names are vastly prefered to short, hard to read names.
    // E.g., use 'first_parameter' or `firstParameter` instead of 'fp'. When in
    // doubt, opt for the longer, easier to read name that conveys the meaning
    // and purpose of the variable. Well named variables are self-documenting.
    // As developers, we usually spend more time reading code than writing code,
    // so the easier you make it to read your code, the more efficient we will
    // all be.

    // DESCRIPTION: In-line comments that describe the logic inside of your
    // functions are extremely helpful both to others as well as your future
    // self to aid in understanding your thought process
  }

 private:
  int my_int;                //< An example private member variable
  std::vector<T> my_vector;  //< An example private member variable
};

// DESCRIPTION: Free functions should be commented in the same way as
// non-trivial class member functions. If the function is templated, use @tparam
// to describe the purpose of the template parameters.
/**---------------------------------------------------------------------------*
 * @brief  An example of a free function (non-class member). This function
 * calls a functor on an input argument and returns the result.
 *
 * @tparam functor_type The type of the functor
 * @tparam input_type The datatype of the input argument
 * @tparam return_type The return type of the functor
 * @param[in] functor The functor to be called on the input argument
 * @param[in] input_argument The input argument passed into the functor
 * @return The result of calling the functor on the input argument
 *---------------------------------------------------------------------------**/
template <class functor_type, typename input_type, typename return_type>
return_type free_function(functor_type functor, input_type input_argument) {
  // Calls the passed in functor on the passed in input argument and returns
  // the result
  return functor(input_argument);
}

// DESCRIPTION: Enumeration types should have a brief overall description of
// the purpose of the enums, as well as a description of each enum member.
/**---------------------------------------------------------------------------*
 * @brief  The purpose of these enumerations is to provide an example
 * of how enumerations should be documented.
 *
 *---------------------------------------------------------------------------**/
enum class example_enum {
  first_enum,   //< Description of the first enum
  second_enum,  //< Description of the second enum
  third_enum    //< Description of the third enum
};
