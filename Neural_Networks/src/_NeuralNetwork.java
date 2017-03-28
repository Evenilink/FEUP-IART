import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import java.util.Arrays;

public class _NeuralNetwork {

    private BackPropagation backPropagation;
    private MultiLayerPerceptron perceptron;

    public _NeuralNetwork(int maxInterations, double maxError, float learningRate) {
        perceptron = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 2, 3, 1);
        backPropagation = new BackPropagation();
        backPropagation.setMaxIterations(maxInterations);
        backPropagation.setMaxError(maxError);
        backPropagation.setLearningRate(learningRate);
    }

    public void learnDataSet(DataSet dataSet) {
        perceptron.learn(dataSet, backPropagation);
        System.out.println("Neural network started learning.");
    }

    public void testNeuralNetwork(NeuralNetwork neuralNetwork, DataSet dataSet) {
        for(DataSetRow row : dataSet.getRows()) {
            neuralNetwork.setInput(row.getInput());
            neuralNetwork.calculate();

            System.out.println("Input: " + Arrays.toString(row.getInput()) +
                    " | Desired output: " + row.getDesiredOutput() +
                    " | Actual output: " + neuralNetwork.getOutput());
        }
    }

    public void saveNeuralNetwork(String filename) {
        perceptron.save(filename);
    }

    public void loadNeuralNetwork(String filename) {
        perceptron = (MultiLayerPerceptron) NeuralNetwork.createFromFile(filename);
    }
}
