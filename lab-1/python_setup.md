

# MATH2071 Lab 1 - Python setup

Matthew Ragoza
2022-01-18

## 1. Unix command line basics

The purpose of this exercise is to gain familiary with basic command line usage. I used a bash terminal on my local machine using Windows Subsystem for Linux.

After running the `pwd` command, the system output the current working directory where my terminal started:
```bash
/mnt/c/Users/mtr22/Code/MATH2071/lab-1
```
Then, running `ls` output a list of all the contents of the current working directory.

I created a new directory in my working directory my running `mkdir test`.

After running `ls` again, the directory I just created appeared in the listed contents of the current working directory, as expected.

Running `cd test` allowed me to move into the directory I just created, setting it as the new working directory.

When I ran `ls` again, there was no output. This is because my current working directory was the new directory I just created, which was empty.

After downloading `hello_world.sh` from Canvas, I put it into the `test` directory and checked that it was there using `ls`.

Then I ran `rm hello_world.sh`, which deleted the file from the current directory. I verified this by running `ls` and noted that the file was no longer listed in the contents.

Next, I moved up to the parent directory using `cd ..`, where the two periods always signify the parent directory of the current path.

Finally, I removed the directory I created previously by running `rmdir test`. When I ran `ls`, the `test` directory was no longer listed in the contents of the current worknig directory, indicating that it had been deleted.

## 2. Visual Studio Code

The purpose of this exercise is to acquire VScode, a popular open-source IDE for software development. I already had VScode installed at the time of starting this lab, so there were no steps to perform.

## 3. Installing Python via Anaconda

The purpose of this exercise is the install Python using the Anaconda platform for package management and scientific computing.

First, I downloaded the installation script for my desired Anaconda version within a bash terminal as follows:
```bash
wget https://repo.continuum.io/archive/Anaconda3-5.3.1-Linux-x86_64.sh
````

I ran the installation script, read the license agreement, and followed the prompts as required. This installed Anaconda and Python within my WSL environment.

## 4. Putting it all together

The purpose of this exercise is to practice using the command line and python, now that we have learned their basic usage.

I started up a new bash terminal, and noticed that it now had a `(base)` tage in the prompt that means that anaconda was working as expected.

I created a folder with `lab01_mattragoza` and a `code` subdirectory of that folder. Then I changed directories into this newly created `code` folder. There, I downloaded the file `lab01.py` from Canvas.

I used the `cat lab01.py` command to display the contents of this python script in the terminal. The script first import that `numpy` module as a variable `np`. Then it assigns integer literal values to two new variables, `x` and `y`. Then it adds them together and assigns their sum to a third variable, `z`. It assign an integer literal to a fourth variable, `t`. Then, the script uses the `zeros` function from numpy to create an array of length 5 filled with zeros, and assign this array to a variable `g`. Finally, the script prints the value of the variable `z` that was created previously to the command line standard output.

After examing the contents of the script, I ran it using `python lab01.py` from within the `code` directory. The output of the command was the following:
```bash
12
```

This is the expected output given the commands in the script. A number of variable assignments and other actions are performed, but python does not produce any outputs by default unless they are explicitly sent to an output stream, for example using the `print` command.

