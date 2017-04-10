import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import java.util.ArrayList;
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
        ArrayList<Double> diffArray = new ArrayList<>();
        System.out.println("Neural network started calculating.");

        for(DataSetRow row : dataSet.getRows()) {
            perceptron.setInput(row.getInput());
            perceptron.calculate();

            double desiredOutput = Double.parseDouble(Arrays.toString(row.getDesiredOutput()).replace("[", "").replace("]", ""));
            double actualOutput = Double.parseDouble(Arrays.toString(perceptron.getOutput()).replace("[", "").replace("]", ""));
            double diff = desiredOutput - actualOutput;
            diffArray.add(diff);

            System.out.println("Desired output: " + desiredOutput + " | Actual output: " + actualOutput + " | Diference: " + diff);
        }

        double diffSum = 0;
        for(int i = 0; i < diffArray.size(); i++)
            diffSum += diffArray.get(i);

        System.out.println("\nAverage difference: " + diffSum / diffArray.size());
    }

    public void SaveNeuralNetwork(String filename) {
        perceptron.save(Utils.TRAINED_NETWORK_FOLDER + filename);
    }

    public void LoadNeuralNetwork(String filename) {
        perceptron = (MultiLayerPerceptron) NeuralNetwork.createFromFile(filename);
    }
}
