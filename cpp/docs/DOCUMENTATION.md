# libcudf++ C++ Documentation Guide

This document is a guide for documenting the libcudf++ source code that is rendered by the
doxygen tool and published to our external RAPIDS [web site](https://docs.rapids.ai/api/libcudf/stable/index.html).

## Copyright License

This is included here but may also be mentioned in a coding guideline document as well.
The following is the license header comment that should appear at the beginning of every C++ source file.

```c++
/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
```

The comment should start with `/*` and not `/**` so it is not processed by doxygen.

Also, here are the rules for the year copyright year.

- A new file should have the year in which it was created
- A modified file should span the year it was created and the year it was modified (e.g. `2019-2020`)

Changing the copyright year may not be necessary if no content has changed (e.g. reformatting only).

## Doxygen

The [doxygen tool](http://www.doxygen.nl/manual/index.html) is used to generate html pages from the C++ comments in the source code. Doxygen will recognize and parse block comments as well as perform specialized output formatting when encountering [doxgen commands](http://www.doxygen.nl/manual/commands.html).

There are almost 200 commands (also called tags in the document) that doxygen recognizes in comment blocks. This document will provide guidance on which commands/tags to use in the libcudf C++ source code.

The doxygen process can be customized using the many, many configuration options in the [Doxyfile](../doxygen/Doxyfile).

Here are some of the custom options in the Doxyfile.
| Option | Setting | Description |
| ------ | ------- | ----------- |
| PROJECT_NAME | libcudf | Title used on the main page |
| PROJECT_NUMBER | 0.14 | Version number |
| EXTENSION_MAPPING | cu=C++ cuh=C++ | Process `cu` and `cuh` as C++ |
| INPUT | main_page.md regex.md unicode.md ../src ../include | Embedded markdown files and source code directories to process |
| FILE_PATTERNS | *.cpp *.hpp *.h *.c *.cu *.cuh | File extensions to process |

## Block Comments

The following block comment style should be used for all C++ doxygen comments.

```c++
/**
 * description text and
 * doxygen tags go here
 */
```

The block should start with `/**` and end with `*/` only and with nothing else on those lines.
(e.g. do not add dashes `-----` or extra asterisks `*****` in these lines).
The block must be placed immediately before the source code line in which it is referring.
The block may be indented to line up vertically with the item they are documenting as appropriate. See the [Example](#the_example) section below.

Each line in the comment block between the `/**` and `*/` lines should start with a space followed by an asterisk. Any text on these lines, including tag declarations, should start after a single space after the asterisk.

## Tag/Command names

Use `@` to prefix doxygen commands (e.g. `@brief`, `@code`, etc.)

## Markdown

The doxygen tool supports a limited set of markdown format in the comment block including links, tables, lists, etc.

In some cases a trade-off may be required for readability in the source text file versus the readability in the doxygen formatted web pages.

For example, there are some limitations on readability with '%' character and pipe character '|' within a table.
TODO: show examples here

Also, try to avoid using direct html tags. Although doxygen supports markdown and markdown supports html tags, the html support for doxygen's markdown is also limited.

## The Example

The following example will cover most of the doxygen block comment and tag styles
for documenting C++ code in libcudf.

```c++
/**
 * @file the_source_file.cpp
 * @brief Description of source file contents
 *
 * Longer description of the source file contents.
 */

/**
 * @brief One sentence description of the class.
 *
 * @ingroup optional_group_name_for_modules_page
 *
 * Longer, more detailed desriptio of the class.
 *
 * @tparam T Short description of any template parameters
 */
template <typename T>
class example_class {

  void get_my_int();            ///< Members can be documented like this
  void set_my_int( int value ); ///< Try to use descriptive names

  /**
   * @brief Short, one sentence description of the member function.
   *
   * A more detailed description of what this function does and what
   * its logic is.
   *
   * @code
   * example_class<int> inst;
   * inst.set_my_int(5);
   * int output = inst.complicated_function(1,dptr,fptr);
   * @endcode
   *
   * @param[in] first  This parameter is an input parameter to the function
   * @param[in,out] second This parameter is used both as an input and output
   * @param[out] third This parameter is an output of the function
   *
   * @return The result of the complex function
   */
  T complicated_function(int first, double* second, float* third)
  {
      // Do not use doxygen-style block comments
      // for code logic documentation.
  }

 private:
  int my_int;                ///< An example private member variable
};

/**
 * @brief Short, one sentence description of this free function.
 *
 * Longer description must start after a blank line.
 *
 * @ingroup optional_one_or_more predefined_groups
 *
 * @code
 * template<typename T>
 * struct myfunctor {
 *   bool operator()(T input) { return input % 2 > 0; }
 * };
 * free_function<myfunctor,int>(myfunctor{},12);
 * @endcode
 *
 * @throw cudf::logic_error if `input_argument` is negative or zero
 *
 * @tparam functor_type The type of the functor
 * @tparam input_type The datatype of the input argument
 *
 * @param[in] functor The functor to be called on the input argument
 * @param[in] input_argument The input argument passed into the functor
 * @return The result of calling the functor on the input argument
 */
template <class functor_type, typename input_type>
bool free_function(functor_type functor, input_type input_argument)
{
  CUDF_EXPECTS( input_argument > 0, "input_argument must be positive");
  return functor(input_argument);
}

/**
 * @brief Short, one sentence description.
 *
 * Optional, longer description.
 */
enum class example_enum {
  first_enum,   ///< Description of the first enum
  second_enum,  ///< Description of the second enum
  third_enum    ///< Description of the third enum
};
```

## Descriptions

The comment description should clearly detail how the output(s) are created from any inputs.
Include any performance and any boundary considerations.
Also include any limits on parameter values and if any default values that are declared.
Don't forget to specify how nulls are handled or produced.
Also, try to include a short [example](#inline_example) if possible.

### @brief

The `@brief` text should be a short, one sentence description.
Doxygen does not provide much space to show this text in the output pages.
Always follow the `@brief` line with a blank comment line.

The longer description is the rest of the comment text that is not tagged with any doxygen command.

### @copydoc

Documentation for declarations in headers is expected to be clear and complete.
You can use the `@copydoc` tag to avoid duplicating the comment block for a function definition for example.

```c++
  /**
   * @copydoc complicated_function(int,double*,float*)
   *
   * Any extra documentation.
   */
```

Also, the `@copydoc` is useful when documenting a `detail` function that differs only by the `cudaStream_t` parameter.

```c++
/**
 * @copydoc cudf::segmented_count_set_bits(bitmask_type const*,std::vector<size_type> const&)
 *
 * @param[in] stream Optional CUDA stream on which to execute kernels
 */
std::vector<size_type> segmented_count_set_bits(bitmask_type const* bitmask,
                                                std::vector<size_type> const& indices,
                                                cudaStream_t stream = 0);
```

Note, you must specify the whole signature of the function, including optional parameters, so that doxygen will be able to locate it.

### Function parameters

The following tags normally appear near the end of function comment block in the following order:

| Command | Description |
| ------- | ----------- |
| @throw | Specify the conditions which the function may throw an exception |
| @tparam | Description for each template parameter |
| @param | Description for each function parameter |
| @return | Short description of object or value returned |

#### @throw

Add an `@throw` comment line in the comment block for each exception that the function may throw.
You only need to include exception thrown by the function itself.
If the function calls another function that may throw an exception, you do not need to document that exception here.

Include the name of the exception without tick marks so doxygen can add reference links correctly.

```c++
 *
 * @throw cudf::logic_error if `input_argument` is negative or zero
 *
```

Using `@throws` is also acceptable but vs-code and other tools only do syntax highlighting on `@throw`.

#### @tparam

Add a `@tparam` comment line for each template parameter declared by this function.
The name of the parameter in the comment must match exactly to the template parameter name.

```c++
 *
 * @tparam functor_type The type of the functor
 * @tparam input_type The datatype of the input argument
 *
```

The definition should detail the requirements of parameter.
For example, if the template is for a functor or predicate, then describe the expected input types and output.

#### @param

Add a `@param` comment line for each function parameter passed to this function.
The name of the parameter in the comment must match the function's parameter name.
Also include append `[in]`, `[out]` or `[in,out]` to the `@param` if it is not clear from the declaration and the parameter name itself.

```c++
 *
 * @param[in] functor The functor to be called on the input argument
 * @param[in] input_argument The input argument passed into the functor
 *
```

#### @return

Add a single `@return` comment line at the end of the comment block if the function returns an object or value.
Include a brief description of what is returned.

```c++
 *
 * @return A new column of type INT32 and no nulls
 */
```

Do not include the type of the object returned with the `@return` comment.

### Inline Examples

It is usually helpful to include a source code example inside your comment block when documenting a function or other declaration.
Use the `@code/@endcode` pair to include inline examples.

Doxygen supports syntax highlighting for C++ and several other programming languages (e.g. Python, Java).
By default, the `@code` tag will use syntax highlighting based on the source code where it was found.

```c++
/**
 * @code
 * auto result = cudf::make_column( );
 * @endcode
 */
```

You can specify a different language by indicating the file extension in the tag:

```c++
/**
 * @code{.py}
 * import cudf
 * s = cudf.Series([1,2,3])
 * @endcode
 */
```

If you wish to use pseudo-code in your example, use the following:

```c++
/**
 * Sometimes pseudo-code is clearer.
 * @code{.pseudo}
 * s = int column of [ 1, 2, null, 4 ]
 * r = fill( s, [1, 2], 0 )
 * r is now [ 1, 0, 0, 4 ]
 * @endcode
 */
```

Do not use the `@example` tag in the comments for a declaration.
Doxygen will interpret the entire source file as example source code when it encounters this tag.
The source file is then published under the 'Examples' tab in the output pages.

## Namespaces

Doxygen output includes a _Namespaces_ page that shows all the namespaces declared with comment blocks in the processed files.
Here is an example of doxygen description comment for a namespace declaration.

```c++
//! cuDF interfaces
namespace cudf {
```

A description comment should be included for only once for each unique namespace declaration.
Otherwise, if more than one description is found, doxygen will aggregate the descriptions in an arbitrary order in the output pages.

## Groups/Modules

Grouping declarations into modules helps users to find APIs in the doxygen pages.
Generally, common functions are already grouped logically into header files but not easily reflected in the doxygen output.
Doxygen output includes a _Modules_ page that organizes items into groups using the [Grouping doxygen commands](http://www.doxygen.nl/manual/grouping.html)
These commands can group common functions across header files, source files, and even namespaces.
Groups can also be nested by defining new groups within existing groups.




## Build Doxygen Output

The doxygen tool can be installed using the instructions on its [Installation](http://www.doxygen.nl/manual/install.html) page.

Building the output is simply running the `doxygen` command while in the `cpp/doxygen` directory containing the `Doxyfile`. The tool will read and process all appropriate source files under the `cpp/` directory. The output will be created in the `cpp/doxygen/html/` directory. You can load the local `index.html` file generated there into any web browser to view the result.

By default, doxygen also uses the `graphviz dot` tool to build some diagrams of the class, namespace, and module relationships. If the `dot` tool cannot be found then the output will be generated without diagrams.

The doxygen installation page does not include instructions for downloading and installing `graphviz dot`.
