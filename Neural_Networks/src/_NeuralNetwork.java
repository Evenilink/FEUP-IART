import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class _NeuralNetwork {

    private BackPropagation backPropagation;
    private MultiLayerPerceptron perceptron;
    private String name;
    private double performance;
    private boolean isLoaded;

    private int maxIterations;
    private double maxError;
    private float learningRate;

    public _NeuralNetwork(String name, int firstLayerCount, int maxIterations, double maxError, float learningRate) {
        this.name = name;
        perceptron = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, firstLayerCount, Utils.NUM_HIDDEN_LAYERS, Utils.NUM_OUTPUT_LAYER);
        SetBackPropagationRules(maxIterations, maxError, learningRate);
        backPropagation = new BackPropagation();
        backPropagation.setMaxIterations(maxIterations);
        backPropagation.setMaxError(maxError);
        backPropagation.setLearningRate(learningRate);
        isLoaded = false;
    }

    public _NeuralNetwork() {
        isLoaded = true;
    }

    public void SetBackPropagationRules(int maxIterations, double maxError, float learningRate) {
        this.maxIterations = maxIterations;
        this.maxError = maxError;
        this.learningRate = learningRate;
    }

    public void LearnDataSet(DataSet dataSet) {
        System.out.println("\tNeural network started learning.");
        perceptron.learn(dataSet, backPropagation);
        System.out.println("\tNeural network has finished learning.\n");
    }

    public void TestNeuralNetwork(DataSet dataSet) throws IOException {
        ArrayList<Double> diffArray = new ArrayList<>();
        System.out.println("\n\tNeural network started calculating.");

        for(DataSetRow row : dataSet.getRows()) {
            perceptron.setInput(row.getInput());
            perceptron.calculate();

            double desiredOutput = Double.parseDouble(Arrays.toString(row.getDesiredOutput()).replace("[", "").replace("]", ""));
            double actualOutput = Double.parseDouble(Arrays.toString(perceptron.getOutput()).replace("[", "").replace("]", ""));
            double diff = actualOutput - desiredOutput;
            diffArray.add(diff);

            System.out.println("\tDesired output: " + desiredOutput + " \t|\t Actual output: " + actualOutput + " \t|\t Difference: " + diff);
        }

        double diffSum = 0;
        for(int i = 0; i < diffArray.size(); i++)
            diffSum += Math.abs(diffArray.get(i));
        performance = diffSum / diffArray.size();

        System.out.println("\n\tAverage difference: " + performance + "\n");
        SaveResultToFile();
    }

    public void SaveNeuralNetwork(String filename) {
        if(name == null)
            name = filename;
        perceptron.save(Utils.TRAINED_NETWORK_FOLDER + filename);
    }

    public void LoadNeuralNetwork(String filename) {
        perceptron = (MultiLayerPerceptron) NeuralNetwork.createFromFile(Utils.TRAINED_NETWORK_FOLDER + filename);
        name = filename;
    }

    public String GetName() {
        return name;
    }

    private void SaveResultToFile() throws IOException {
        // If the network was loaded, there's no need to write to the file, since it's already been tested.
        if(isLoaded)
            return;

        // Creates file if it doesn't exists.
        File file = new File(Utils.PERFORMANCE_FOLDER + name + ".txt");
        file.createNewFile();

        String str = maxIterations + " " + maxError + " " + learningRate + " " + performance;


        FileReader fr = new FileReader(Utils.PERFORMANCE_FOLDER + name + ".txt");
        BufferedReader br = new BufferedReader(fr);
        String line;
        int i = 0;

        // Detects where to put the current back propagation rules in the file, since the top most rules are the best for this network.
        while((line = br.readLine()) != null) {
            String[] msgSplit = line.split(" ");
            double filePerformance = Double.parseDouble(msgSplit[3]);

            if(performance < filePerformance)
                break;
            i++;
        }

        br.close();
        fr.close();

        Path path = Paths.get(Utils.PERFORMANCE_FOLDER + name + ".txt");
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        lines.add(i, str);
        Files.write(path, lines, StandardCharsets.UTF_8);
    }

    @Override
    public boolean equals(Object obj) {
        if(obj instanceof _NeuralNetwork) {
            _NeuralNetwork nn = (_NeuralNetwork) obj;
            return (nn.GetName().equals(name));
        }
        return false;
    }
}