# Image_Text_Recognition_System

In this problem, we were provided with the following :

* train-text.txt - a training dataset and a text document that is representative of the English language. It is useful 
for the purpose of training our classifier with some English language text.

* train-image-file.png - The train-image-File.png contains a perfect (noise-free) version of each letter that the code uses to 
test and compare the test image.

* test-image-file.png - this file contains a slightly distorted and noise filled image of text, that our program should decode and 
return the actual wordings.

* image2text.py - it contained the skeleton code where we are supposed to publish our solution.

## Problem Statement

To write a program that recognizes text in an image. But the images are noisy, so any particular letter may be difficult to recognize. However, if we make the assumption that these images have English words and sentences, we can use statistical properties of the language to resolve ambiguities.

## Solution

In our solution to the problem described above, we've used two different approaches to tackle the problem. Both the approaches are defined below 

### Simple Approach

Here, first the training image was loaded and decoded into an array of fixed length consisting of "," and "*". This helped in categorizing every element of our training picture uniquely. In the next step, same was done with the testing image, difference being that the letters of test image were unknown. After these conversions, a simple classifier was used where each element of testing data was compared with each element of the training data (pixel matching), and 
based on a hit or a miss, certain weight was assigned to it, in order to get the maximum probability value whenever a match is encountered. As a result, the comparison with the maximum number of matches, resulted in the maximum value and therefore after every iteration, the letter with the maximum value was pushed into an output string, which was our final output for this process.

### HMM Approach

In order to implement HMM approach with viterbi, we were responsible for calculating a number of probabilistic values - initial probability, transmission probability and emission probability. Emission probability was calculated the exact same way as we calculated probability during our Simple Bayes Net approach (pixel comparison). Initial probability was calculated using the frequency of the first letter of every line in our training file and dividing it by the total words
in our training file. And Lastly, transmission probability was calculated in a similar fashion, only difference being that in place of just the first character of every word, we took the combination of two consecutive characters and then computed the probability based on the frequency. 

Once we had these probabilities value, we have used the viterbi formula to calculate the final probability value for every image letter. Here, we have used the negative log of every value and considered the minimum value of those ( reference taken from the method explained in the class). This was done as the probability value was coming out to be very low after continuous multiplication.  

## Design Decisions

A number of data structures including list, dictionary etc. and a variety of inbuilt functions like math.log, min, max were used in this solution. The main approach to decide the designing decisions for this problem was to implement the solution in minimum time complexity. Log of probabilities was taken and added in place of multiplying so that heavy computation is avoided and to ensure that very small probabilistic values are not ignored while calculating the final probability. 

## Assumpltions

Two major assumptions were done in this problem to ease the problem and make the complexity a bit low. Firstly, it was assumed that all the text in our images has the same fixed-width font of the same size. Secondly, we'll also assume that our documents only have the 26 uppercase latin characters, the 26 lowercase characters,
the 10 digits, spaces, and 7 punctuation symbols, (),.-!?'".

## Challenges Faced

A number of problems were encountered while designing the solution. A few of them are enlisted below.

* As our probabilistic values was highly dependent on the frequency of characters in our training dataset. If there was a case when a particular character was not found in our training dataset but was present in our testing image, then that resulted in probability value to be zero. To make this work a very small buffer value was added to every character so that it never becomes zero.

* Lastly, assigning specific weights while calculating the emission probability was needed to be accurate so that the precision is achieved by the system.
