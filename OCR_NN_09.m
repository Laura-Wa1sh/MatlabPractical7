clear all
close all

%% training Stage

% Loading labels and examples of handwritten digits from MNIST Dataset
%Since they are too many of them, to make it usable for this practical we
%will be taking one out of many images (one out of 'sampling'). This is
%done by using the variable sampling>1
sampling=10;
images = loadMNISTImages('train-images',sampling);
labels = loadMNISTLabels('train-labels',sampling);

% For visualization purposes, we display the first 100 images
figure
for i=1:100

    % As you can notice by the size of the matrix image, each digit image
    % has been transform into a long feature vector to be fed in a machine
    % learning algorithm.
    
    %To visualise or recompose the image again, we need to revert that
    %process in its 28x28 image format
    Im = reshape(images(i,:),28,28);
    subplot(10,10,i), imshow(Im), title(['label: ',num2str(labels(i))])
    
end

%[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis (images);
[eigenVectors, eigenValues, meanX, Xlda] = LDA(labels,[],images);

%Supervised training function that takes the examples and infers a model
modelNN = NNtraining(Xlda, labels);


%% testing
% Loading testing labels and testing examples of handwritten digits from MNIST Dataset
% It is very important that this images are different from the ones used in
% training or our results will not be reliable

images = loadMNISTImages('test-images',sampling);
labels = loadMNISTLabels('test-labels',sampling);

confusionMatrix = zeros(10);
noOfInstances = zeros(10,1);


%For each testing image, we obtain a prediction based on our trained model
for i=1:size(images,1)
    
    testnumber= images(i,:);
    
    Xpca = (testnumber - meanX)* eigenVectors;
    
    classificationResult(i,1) = KNNTesting(Xpca, modelNN, 3);
    
    confusionMatrix(labels(i,1)+1,classificationResult(i,1)+1) = confusionMatrix(labels(i,1)+1,classificationResult(i,1)+1) +1;
    noOfInstances(labels(i,1)+1,1) = noOfInstances(labels(i,1)+1,1) +1;
    
end


%% Evaluation

% Finally we compared the predicted classification from our mahcine
% learning algorithm against the real labelling of the esting image
comparison = (labels==classificationResult);

for i=1:size(confusion_matrix)
    for j=1:size(confusion_matrix)
    confusion_matrix(i,j) = confusion_matrix(i,j)/ numberOfInstances(i)
    end
end

%Accuracy is the most common metric. It is defiend as the numebr of
%correctly classified samples/ the total number of tested samples
Accuracy = sum(comparison)/length(comparison)


%We display 100 of the correctly classified images
figure
title('Correct Classification')
count=0;
i=1;
while (count<100)&&(i<=length(comparison))
   
    if comparison(i)
        count=count+1;
        subplot(10,10,count)
        Im = reshape(images(i,:),28,28);
        imshow(Im)
    end
    
    i=i+1;
    
end


%We display 100 of the incorrectly classified images
figure
title('Wrong Classification')
count=0;
i=1;
while (count<100)&&(i<=length(comparison))
    
    if ~comparison(i)
        count=count+1;
        subplot(10,10,count)
        Im = reshape(images(i,:),28,28);
        imshow(Im)
        title(num2str(classificationResult(i)))
    end
    
    i=i+1;
    
end