# Measuring Influence

Let's say we have a function f(x). 

>Y = f(x) (say)

Now we know that dY/dx is proportional to:

>f(x+h) - f(x)

Now, what do I exactly mean by 'influential' here? 

By x is 'influential' to f(x), I mean, f(x) changes rather greatly (at least compared to the case where x is less influential) with change in the value of x. 

And, x has little 'influence' on f(x), when f(x) barely changes even when x greatly changes. 

Since dy/dx is basically the measure of change in y or f(x) with respect to a negligible change in x,
|dy/dx| should usually be greater for cases where x is more influential to f(x).

(Note: I am ignoring the direction of change, and focusing on the magnitude; thus |dy/dx| instead of dy/dx)

Now, if we have a function 

>y = f(x1,x2,x3), 

we can compare the individual 'influences' of x1,x2,x3 on f(x1,x2,x3) by
comparing the values of 

>|pd(y)/pd(x1)|, |pd(y)/pd(x2)|, and |pd(y)/pd(x3)| 

(where pd represents partial differentiation).

# Finding Influential Input Features

Now for an image classification neural model, we can consider y = h(x1,x2,x3....xn) as the representative function for the neural network. Here, x1,x2,x3....xn are input pixels of the images to be classified. 

Now we can compute the gradients |pd(h(x1,x2,x3...xn))/pd(xi)| for each input features. 

From there, we should be able to select some of the most influential input features i.e. the features for whose
the value of the gradients are higher than most of the others. 

# Object Discovery ?

Let's say we have an input image of a dog. The classifier network classifies that image to be a dog. Now intuitively, the most influential pixels of that image should be the ones near the object dog within the image. If the classifier is working properly, surely it should be the pixels that constitutes the 'dog' portion of the image, due to which the classifier classifies the image to have a dog. 

If that's true then according to the theory, we can find the approximately most influential pixels simply by computing the gradients as described before. 

Now, if the theory is right, we will end up finding the pixels that are near about the object-dog. If indeed that happens, we can acheive some level of object discovery without any further supervised or unsupervised learning; just by extracting latent learned information from a network trained in classification.

We might be even able to extrapolate the data and even create bounding boxes, or something like that, and acheive full on object detection. 

By modifying to network function by multiplying the output with some masking matrix such the result becomes the value of only one specific class, we may be even able to find pixels that specifically influences the certain class. 

# Implementation

I am working on a pre-trained model ([wide-residual-network](https://github.com/JRC1995/Wide-Residual-Network))

The data pre-processing script, the saved models and all else are available here: https://github.com/JRC1995/Wide-Residual-Network

This implementation presented here is basically a toy implementation.

Ideally, I wanted to compute the gradients of the final model output w.r.t the input image data, however doing so resulted in gradient explosion. Normally in training, gradient doesn't explodes when batch normalization training phase is set as true. But while making predictions on single data, when batch normalization training phase is set as false, gradients explode.

So I only calculated the gradients of the output of the second convolutional block w.r.t the image data. 

I then sorted the absolute values of the gradients, chose a threshold value, and then marked all input pixels beyond the threshold as black, and all else as white.

We can <b>try to consider the positions of the black pixels to be trying to representing the positions of the pixels of the actual classification-object</b>, and from that perspective, we can subjectively evaluate the results.  

So, overall, this implementation is pretty rough around the edges: it's incomplete and also isn't evaluated with any objective metric.

This method, however, can also bring some layer interpretability. 

Here are some results:

![png](/Images/output_4_1.png)


    After Processing: 



![png](/Images/output_4_3.png)


    100 Most influential pixels (in black):



![png](/Images/output_4_5.png)


    
    There's about a 99.831% chance that there is at least one bird in the image
    
    The whole probability distribution:
    
    airplane: 0.148%
    automobile: 0.000%
    bird: 99.831%
    cat: 0.000%
    deer: 0.020%
    dog: 0.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.001%
    truck: 0.000%
    
    Enter relative path to the image: car3.jpg



![png](/Images/output_4_7.png)


    After Processing: 



![png](/Images/output_4_9.png)


    100 Most influential pixels (in black):



![png](/Images/output_4_11.png)


    
    There's about a 54.035% chance that there is at least one frog in the image
    
    The whole probability distribution:
    
    airplane: 13.371%
    automobile: 27.672%
    bird: 0.042%
    cat: 4.181%
    deer: 0.001%
    dog: 0.000%
    frog: 54.035%
    horse: 0.008%
    ship: 0.000%
    truck: 0.689%
    
    Enter relative path to the image: bird.jpg



![png](/Images/output_4_13.png)


    After Processing: 



![png](/Images/output_4_15.png)


    100 Most influential pixels (in black):



![png](/Images/output_4_17.png)


    
    There's about a 100.000% chance that there is at least one bird in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 100.000%
    cat: 0.000%
    deer: 0.000%
    dog: 0.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: ls.jpg



![png](/Images/output_4_19.png)


    After Processing: 



![png](/Images/output_4_21.png)


    100 Most influential pixels (in black):



![png](/Images/output_4_23.png)


    
    There's about a 99.854% chance that there is at least one dog in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.105%
    cat: 0.041%
    deer: 0.000%
    dog: 99.854%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: cat.jpg



![png](/Images/output_4_25.png)


    After Processing: 



![png](/Images/output_4_27.png)


    100 Most influential pixels (in black):



![png](/Images/output_4_29.png)


    
    There's about a 99.253% chance that there is at least one cat in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.282%
    cat: 99.253%
    deer: 0.000%
    dog: 0.001%
    frog: 0.000%
    horse: 0.463%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: car.jpg



![png](/Images/output_4_31.png)


    After Processing: 



![png](/Images/output_4_33.png)


    100 Most influential pixels (in black):



![png](/Images/output_4_35.png)


    
    There's about a 58.512% chance that there is at least one bird in the image
    
    The whole probability distribution:
    
    airplane: 16.885%
    automobile: 0.796%
    bird: 58.512%
    cat: 19.012%
    deer: 4.099%
    dog: 0.000%
    frog: 0.116%
    horse: 0.061%
    ship: 0.107%
    truck: 0.413%
    
    Enter relative path to the image: smalltruck.jpg



![png](/Images/output_4_37.png)


    After Processing: 



![png](/Images/output_4_39.png)


    100 Most influential pixels (in black):



![png](/Images/output_4_41.png)


    
    There's about a 99.884% chance that there is at least one truck in the image
    
    The whole probability distribution:
    
    airplane: 0.105%
    automobile: 0.011%
    bird: 0.000%
    cat: 0.000%
    deer: 0.000%
    dog: 0.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 99.884%
    
    Enter relative path to the image: STOP







