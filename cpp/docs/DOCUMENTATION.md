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
 * ... text ...
 */
```

The block should start with `/**` and end with `*/` only and with nothing else on those lines.
(e.g. do not add dashes `-----` or extra asterisks `*****` in these lines).
The block must be placed immediately before the source code line in which it is referring.
The block may be indented to line up vertically with the item they are documenting as appropriate. See the [Example](#the_example) section below.

## Tag/Command names

Use `@` to prefix doxygen commands (e.g. `@brief`, `@code`, etc.)

## Markdown

The doxygen tool supports a limited set of markdown format in the comment block including links, tables, lists, etc.

In some cases a trade-off may be required for readability in the source text file versus the readability in the doxygen formatted web pages.

For example, there are some limitations on readability with '%' character and pipe character '|' within a table.
TODO: show examples here

Also, try to avoid using direct html tags. Although doxygen supports markdown and markdown supports html tags, the html support for doxygen's markdown is also limited.

### Including markdown file

TODO: describe including markdown file with `@include` or in Doxyfile

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

The comment description should clearly define how the output(s) are
created from any inputs. 
Don't forget to include how nulls are handled.
Also, try to include a short example if possible.
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

### @brief

The `@brief` text should be a short, one sentence description. Doxygen does not provide much space to show this text. Always follow the `@brief` line with a blank comment line.

The longer description is the remainder of the comment text that is not tagged as
a doxygen command.

### @copydoc

Documentation for declarations in headers is expected to be clear and complete. There is no reason to duplicate the comment block for a function definition.

TODO: show example

Also, this is useful when documenting a `detail` function that differs only by the `cudaStream_t` parameter.

TODO: show example

Note, you must include the whole signature of the function so that doxygen will be able to locate it.

### Function parameters (@throw/@tparam/@param/@return)

The following tags normally appear near the end of comment block in the following order:

| Command | Description |
| ------- | ----------- |
| @throw | Include the name of the exception without tick marks |
| @tparam | Template parameter |
| @param | Function parameter names must match. Also include `[in]`, `[out]` or `[in|out]` if it is not clear from the declaration and the name |
| @return | Short description of object returned |

## Inline Examples

It is usually helpful to include a source code example inside your comment block when documenting a function or other declaration.

Use the `@code/@endcode` pair to include inline examples.

Doxygen supports syntax highlighting for C++ and several other programming languages (e.g. Python, Java).
By default, the @code tag will use syntax highlighting based on the source code where it was found.

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

Do not use the `@example` tag in the comments for a declaration.
Doxygen will interpret the entire source file as example source code when it encounters this tag.
The source file is then published under the 'Examples' tab in the output pages.

## Groups/Modules

TODO

## Build Output

The doxygen tool can be installed using the instructions on their [Installation](http://www.doxygen.nl/manual/install.html) page.


Building the output is simply running the `doxygen` command while in the `cpp/doxygen` directory containing the Doxyfile. The tool will read and process all appropriate source files under the `cpp/` directory. The output will be created in the `cpp/doxygen/html/` directory. You can load the `index.html` file generated there into any web browser to view the result.

By default, doxygen also employs the `graphviz dot` tool to build some diagrams of the class, namespace, and module relationships. If the `dot` tool cannot be found then the output will be generated without diagrams.

The doxygen installation page does not include instructions for downloading and installing `graphviz dot`.
