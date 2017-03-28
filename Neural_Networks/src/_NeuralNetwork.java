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

    public _NeuralNetwork(int maxInterations, double maxError, float learningRate, int firstLayerCount) {
        perceptron = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, firstLayerCount, 350, 1);
        backPropagation = new BackPropagation();
        backPropagation.setMaxIterations(maxInterations);
        backPropagation.setMaxError(maxError);
        backPropagation.setLearningRate(learningRate);
        System.out.println("Neural network ready.");
    }

    public void learnDataSet(DataSet dataSet) {
        System.out.println("Neural network started learning.");
        perceptron.learn(dataSet, backPropagation);
    }

    public void testNeuralNetwork(DataSet dataSet) {
        System.out.println("Neural network started calculating.");

        for(DataSetRow row : dataSet.getRows()) {
            perceptron.setInput(row.getInput());
            perceptron.calculate();

            System.out.println("Desired output: " + Arrays.toString(row.getDesiredOutput()) +
                    " | Actual output: " + Arrays.toString(perceptron.getOutput()));
        }
    }

    public void saveNeuralNetwork(String filename) {
        perceptron.save(filename);
    }

    public void loadNeuralNetwork(String filename) {
        perceptron = (MultiLayerPerceptron) NeuralNetwork.createFromFile(filename);
    }
}
