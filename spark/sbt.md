

# Basics

## basic project

```
$ mkdir hello
$ cd hello
$ echo 'object Hi { def main(args: Array[String]) = println("Hi!") }' > hw.scala
$ sbt
...
> run
...
Hi!
```

In this case, sbt works purely by convention. sbt will find the following automatically:

  - Sources in the base directory
  - Sources in src/main/scala or src/main/java
  - Tests in src/test/scala or src/test/java
  - Data files in src/main/resources or src/test/resources
  - jars in lib

## Build definition 

Most projects will need some manual setup. Basic build settings go in a file called build.sbt, located in the project’s base directory.

For example, if your project is in the directory hello, in hello/build.sbt you might write:

```
lazy val root = (project in file(".")).
  settings(
    name := "hello",
    version := "1.0",
    scalaVersion := "2.11.7"
  )
```
If you plan to package your project in a jar, you will want to set at least the name and version in a build.sbt.

## Directory structure

sbt uses the same directory structure as Maven for source files by default (all paths are relative to the base directory):

```
src/
  main/
    resources/
       <files to include in main jar here>
    scala/
       <main Scala sources>
    java/
       <main Java sources>
  test/
    resources
       <files to include in test jar here>
    scala/
       <test Scala sources>
    java/
       <test Java sources>
```
Other directories in src/ will be ignored. Additionally, all hidden directories will be ignored

You’ve already seen *build.sbt* in the project’s base directory. Other sbt files appear in a project subdirectory. project can contain *.scala* files, which are combined with *.sbt* files to form the complete build definition. See organizing the build for more.

```
build.sbt
project/
  Build.scala
```

You may see *.sbt* files inside *project/* but they are not equivalent to *.sbt* files in the project’s base directory. Explaining this will come later, since you’ll need some background information first.
Build products 

Generated files (compiled classes, packaged jars, managed files, caches, and documentation) will be written to the **target** directory by default.

## Running

Interactive mode 

Run sbt in your project directory with no arguments:

$ sbt

Running sbt with no command line arguments starts it in interactive mode. Interactive mode has a command prompt (with tab completion and history!).

For example, you could type compile at the sbt prompt: `> compile`. To compile again, press up arrow and then enter. To run your program, type `run`. To leave interactive mode, type exit or use Ctrl+D (Unix) or Ctrl+Z (Windows).

You can also run sbt in **batch mode**, specifying a space-separated list of sbt commands as arguments. For sbt commands that take arguments, pass the command and arguments as one argument to sbt by enclosing them in quotes. For example, `$ sbt clean compile "testOnly TestA TestB"`

In this example, testOnly has arguments, TestA and TestB. The commands will be run in sequence (clean, compile, then testOnly).


To speed up your edit-compile-test cycle, you can ask sbt to **automatically recompile or run tests** whenever you save a source file. Make a command run when one or more source files change by prefixing the command with **~**. For example, in interactive mode try: `> ~ compile` . Press enter to stop watching for changes. You can use the ~ prefix with either interactive mode or batch mode.

Here are some of the most common sbt commands. For a more complete list, see Command Line Reference.
 - **clean** Deletes all generated files (in the target directory).
 - **compile** Compiles the main sources (in src/main/scala and src/main/java directories).
 - **test** Compiles and runs all tests.
 - **console** Starts the Scala interpreter with a classpath including the compiled sources and all dependencies. To return to sbt, type :quit, Ctrl+D (Unix), or Ctrl+Z (Windows).
 - **run <argument>** Runs the main class for the project in the same virtual machine as sbt.
 - **package** Creates a jar file containing the files in src/main/resources and the classes compiled from src/main/scala and src/main/java.
 - **help <command>** Displays detailed help for the specified command. If no command is provided, displays brief descriptions of all commands.
 - **reload** Reloads the build definition (build.sbt, project/*.scala, project/*.sbt files). Needed if you change the build definition.


# References

 - [sbt-0.13 doc](http://www.scala-sbt.org/0.13/docs/Basic-Def.html)