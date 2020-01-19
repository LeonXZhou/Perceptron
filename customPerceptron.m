%% Custom Perceptron
% Assignment 1 CMPE 452 by Leon Zhou 10177892
% 2 layer neural network using 8 input nodes and 3 output nodes. 
% 1 input node is a bias the other 7 represent area, perimeter, compactness,
% length, width, asymmetry coefficient, length of kernel groove
% the output nodes represent the 3 possible wheat seeds classes (kama, rosa,
% canadian) the highest output node represents the predicted class
% the network is trained using simple feedback method with a termination
% condition of 97% of training data being correctly classified
%% read data and format data
fileID = fopen('textfile.txt', 'a');
trainData = csvread('trainSeeds.csv');
inputData = trainData(:,1:7);
bias = ones(length(inputData),1);
expected = trainData(:,8);

%concatenate bias and inputData values into one matrix
inputValues = [inputData,bias];
weights = rand(8,3);
formatSpec = '%4.2f,';
fprintf(fileID,'initial weights for neuron 1:');
fprintf(fileID,formatSpec,weights(:,1));
fprintf(fileID,'\ninitial weights for neuron 2:');
fprintf(fileID,formatSpec,weights(:,2));
fprintf(fileID,'\ninitial weights for neuron 3:');
fprintf(fileID,formatSpec,weights(:,3));

c = 0.000001; %training rate

%% train
counter = 0; %% keeps track of number of correctly classified wheat seeds per iteration
iteration = 0;
while(counter < .97*165)
    counter = 0; 
    iteration = iteration+1;
    for i = 1:165
        predicted = inputValues(i,:)*weights; % calculates the prediction matrix.
                                              % max index represents the
                                              % predicted class. index 1
                                              % is kama, index 2 is rosa,
                                              % index 3 is canadian
        if (expected(i) == 1)
            % if the expected class is kama adjust weights leading into
            % neuron 1 to produce values closer to 1 and weights leading
            % into neuron 2 and 3 to be closer to 0
            if (predicted(1) > 1)
                weights(:,1) = weights(:,1)-c.*inputValues(i,:)';
            elseif (predicted(1) <1)
                weights(:,1) = weights(:,1)+c.*inputValues(i,:)';
            end
            
            if (predicted(2) > 0)
                weights(:,2) = weights(:,2)-c.*inputValues(i,:)';
            elseif (predicted(2) <0)
                weights(:,2) = weights(:,2)+c.*inputValues(i,:)';
            end

            if (predicted(3) > 0)
                weights(:,3) = weights(:,3)-c.*inputValues(i,:)';
            elseif (predicted(3) < 0)
                weights(:,3) = weights(:,3)+c.*inputValues(i,:)';
            end
            % if index 1 in prediction is highest this means the predicted
            % wheat seeds is correct therefore increment counter
            if (predicted(1)>predicted(2) && predicted(1)>predicted(3))
                counter = counter+1;
            end
        end
        if (expected(i) == 2)
            % if the expected class is kama adjust weights leading into
            % neuron 2 to produce values closer to 1 and weights leading
            % into neuron 1 and 3 to be closer to 0
            if (predicted(1) > 0)
                weights(:,1) = weights(:,1)-c.*inputValues(i,:)';
            elseif (predicted(1) <0)
                weights(:,1) = weights(:,1)+c.*inputValues(i,:)';
            end

            if (predicted(2) > 1)
                weights(:,2) = weights(:,2)-c.*inputValues(i,:)';
            elseif (predicted(2) <1)
                weights(:,2) = weights(:,2)+c.*inputValues(i,:)';
            end

            if (predicted(3) > 0)
                weights(:,3) = weights(:,3)-c.*inputValues(i,:)';
            elseif (predicted(3) < 0)
                weights(:,3) = weights(:,3)+c.*inputValues(i,:)';
            end
            % if index 2 in prediction is highest this means the predicted
            % wheat seeds is correct therefore increment counter
            if (predicted(2)>predicted(1) && predicted(2)>predicted(3))
                counter = counter+1;
            end
        end

        if (expected(i) == 3)
            % if the expected class is canadian adjust weights leading into
            % neuron 3 to produce values closer to 1 and weights leading
            % into neuron 2 and 1 to be closer to 0
            if (predicted(1) > 0)
                weights(:,1) = weights(:,1)-c.*inputValues(i,:)';
            elseif (predicted(1) <0)
                weights(:,1) = weights(:,1)+c.*inputValues(i,:)';
            end

            if (predicted(2) > 0)
                weights(:,2) = weights(:,2)-c.*inputValues(i,:)';
            elseif (predicted(2) <0)
                weights(:,2) = weights(:,2)+c.*inputValues(i,:)';
            end

            if (predicted(3) > 1)
                weights(:,3) = weights(:,3)-c.*inputValues(i,:)';
            elseif (predicted(3) < 1)
                weights(:,3) = weights(:,3)+c.*inputValues(i,:)';
            end
            % if index 3 in prediction is highest this means the predicted
            % wheat seeds is correct therefore increment counter
            if (predicted(3)>predicted(2) && predicted(3)>predicted(1))
                counter = counter+1;
            end
        end
    end
end
%% Test
testData = csvread('testSeeds.csv');
inputData = testData(:,1:7);
bias = ones(length(inputData),1);
expected = testData(:,8);
%concatenate bias and inputData values into one matrix
inputValues = [inputData,bias];
%compare predictions to test data.
predictedValues = zeros(length(testData),1);
for i = 1:length(testData)  
    prediction = inputValues(i,:)*weights;
    [M,I] =  max(prediction);
    predictedValues(i) = I;
end
formatSpec = '%4.2f,';
fprintf(fileID,'\nFinal weights for neuron 1:');
fprintf(fileID,formatSpec,weights(:,1));
fprintf(fileID,'\nFinal weights for neuron 2:');
fprintf(fileID,formatSpec,weights(:,2));
fprintf(fileID,'\nFinal weights for neuron 3:');
fprintf(fileID,formatSpec,weights(:,3));

fprintf(fileID,'\ntotal iterations used:%1d\n',iteration);
fprintf(fileID,'Predicted and expected values of test data:\n');
formatSpec = 'Expected: %1d Predicted: %1d\n';
fprintf(fileID,formatSpec,expected,predictedValues);
confusionmat(expected,predictedValues)
fclose(fileID);
    



