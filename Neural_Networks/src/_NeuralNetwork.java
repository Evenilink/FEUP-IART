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
    private DataSet learningDateSet;
    private double performance;
    private boolean isLoaded;

    private int maxIterations;
    private double maxError;
    private float learningRate;
    private int hiddenNodesCount;
    private float learnTime;

    public _NeuralNetwork(String name, int inputNodesAmount, int hiddenNodesCount, int maxIterations, double maxError, float learningRate) throws IOException {
        this.name = name;
        perceptron = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, inputNodesAmount, hiddenNodesCount, Utils.NUM_OUTPUT_NODES);
        SetBackPropagationRules(hiddenNodesCount, maxIterations, maxError, learningRate);
        backPropagation = new BackPropagation();
        backPropagation.setMaxIterations(maxIterations);
        backPropagation.setMaxError(maxError);
        backPropagation.setLearningRate(learningRate);
        isLoaded = false;
    }

    public _NeuralNetwork() {
        isLoaded = true;
    }

    public void SetBackPropagationRules(int hiddenNodesCount, int maxIterations, double maxError, float learningRate) {
        this.hiddenNodesCount =hiddenNodesCount;
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.maxError = maxError;
    }

    public void LearnDataSet(DataSet dataSet, boolean displayResults) {
        if(displayResults)
            System.out.println("\tNeural network started learning.");

        long timeStart = System.currentTimeMillis();
        perceptron.learn(dataSet, backPropagation);
        learnTime = (System.currentTimeMillis() - timeStart) / 1000f;

        if(displayResults)
            System.out.println("\tNeural network has finished learning.\n");
    }

    public void TestNeuralNetwork(DataSet dataSet, boolean displayResults) throws IOException {
        ArrayList<Double> diffArray = new ArrayList<>();

        if(displayResults)
            System.out.println("\n\tNeural network started calculating.");

        for(DataSetRow row : dataSet.getRows()) {
            perceptron.setInput(row.getInput());
            perceptron.calculate();

            double desiredOutput = Double.parseDouble(Arrays.toString(row.getDesiredOutput()).replace("[", "").replace("]", ""));
            double actualOutput = Double.parseDouble(Arrays.toString(perceptron.getOutput()).replace("[", "").replace("]", ""));
            double diff = actualOutput - desiredOutput;
            diffArray.add(diff);

            if(displayResults)
                System.out.println("\tDesired output: " + desiredOutput + " \t|\t Actual output: " + actualOutput + " \t|\t Difference: " + diff);
        }

        double diffSum = 0;
        for(int i = 0; i < diffArray.size(); i++)
            diffSum += Math.abs(diffArray.get(i));
        performance = diffSum / diffArray.size();

        if(displayResults)
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

        String str = hiddenNodesCount + " " + maxIterations + " " + maxError + " " + learningRate + " " + performance + " " + learnTime;

        FileReader fr = new FileReader(Utils.PERFORMANCE_FOLDER + name + ".txt");
        BufferedReader br = new BufferedReader(fr);
        String line;
        int i = 0;

        boolean replace = false;
        // Detects where to put the current back propagation rules in the file, since the top most rules are the best for this network.
        while((line = br.readLine()) != null) {
            String[] msgSplit = line.split(" ");
            int fileHiddenNodesAmount = Integer.parseInt(msgSplit[0]);
            int fileMaxIterations = Integer.parseInt(msgSplit[1]);
            double fileMaxError = Double.parseDouble(msgSplit[2]);
            float fileLearningRate = Float.parseFloat(msgSplit[3]);
            double filePerformance = Double.parseDouble(msgSplit[4]);

            // If the rules are the same but the performance is higher or equal, there's no need to write to the file.
            if(fileHiddenNodesAmount == hiddenNodesCount &&
                    fileMaxIterations == maxIterations &&
                    fileMaxError == maxError &&
                    fileLearningRate == learningRate) {
                if(performance < filePerformance) {
                    replace = true;
                    break;
                } else
                    return;
            }

            if(performance < filePerformance)
                break;
            i++;
        }

        br.close();
        fr.close();

        Path path = Paths.get(Utils.PERFORMANCE_FOLDER + name + ".txt");
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        lines.add(i, str);
        if(replace)
            lines.remove(i+1);
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