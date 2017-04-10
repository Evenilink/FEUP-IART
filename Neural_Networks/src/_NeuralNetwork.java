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

    public _NeuralNetwork(int firstLayerCount) {
        perceptron = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, firstLayerCount, Utils.NUM_HIDDEN_LAYERS, Utils.NUM_OUTPUT_LAYER);
        backPropagation = new BackPropagation();
        backPropagation.setMaxIterations(Utils.MAX_ITERATIONS);
        backPropagation.setMaxError(Utils.MAX_ERROR);
        backPropagation.setLearningRate(Utils.LEARNING_RATE);
        System.out.println("Neural network ready.");
    }

    public void LearnDataSet(DataSet dataSet) {
        System.out.println("Neural network started learning.");
        perceptron.learn(dataSet, backPropagation);
    }

    public void TestNeuralNetwork(DataSet dataSet) {
        System.out.println("Neural network started calculating.");

        for(DataSetRow row : dataSet.getRows()) {
            perceptron.setInput(row.getInput());
            perceptron.calculate();

            System.out.println("Desired output: " + Arrays.toString(row.getDesiredOutput()) +
                    " | Actual output: " + Arrays.toString(perceptron.getOutput()));
        }
    }

    public void SaveNeuralNetwork(String filename) {
        perceptron.save(Utils.TRAINED_NETWORK_FOLDER + filename);
    }

    public void LoadNeuralNetwork(String filename) {
        perceptron = (MultiLayerPerceptron) NeuralNetwork.createFromFile(filename);
    }
}
