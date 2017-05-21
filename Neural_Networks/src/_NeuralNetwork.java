import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.error.ErrorFunction;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.data.sample.SubSampling;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class _NeuralNetwork {

    private BackPropagation backPropagation;
    private MultiLayerPerceptron perceptron;
    private String name;
    private DataSet learningDateSet;
    private DataSet testDataSet;
    private boolean isLoaded;

    private int maxIterations;
    private double maxError;
    private float learningRate;
    private float learnTime;

    private double networkError;

    /**
     * Neural network creation constructor.
     * @param name
     * @param hiddenLayers
     * @param maxIterations
     * @param maxError
     * @param learningRate
     * @throws IOException
     */
    public _NeuralNetwork(String name, ArrayList<Integer> hiddenLayers, int maxIterations, double maxError, float learningRate) throws IOException {
        this.name = name;

        ArrayList<Integer> nodesInLayers = new ArrayList<>();
        nodesInLayers.add(Utils.NUM_INPUT_NODES);
        for(int i = 0; i < hiddenLayers.size(); i++)
            nodesInLayers.add(hiddenLayers.get(i));
        nodesInLayers.add(Utils.NUM_OUTPUT_NODES);

        perceptron = new MultiLayerPerceptron(nodesInLayers, TransferFunctionType.SIGMOID);

        SetBackPropagationRules(maxIterations, maxError, learningRate);
        backPropagation = new BackPropagation();
        backPropagation.setNeuralNetwork(perceptron);
        backPropagation.setMaxIterations(maxIterations);
        backPropagation.setMaxError(maxError);
        backPropagation.setLearningRate(learningRate);
        isLoaded = false;
        CreateDataSets(name);
    }

    /**
     * Neural Network loading constructor.
     * @param name
     * @throws IOException
     */
    public _NeuralNetwork(String name) throws IOException {
        isLoaded = true;
        perceptron = (MultiLayerPerceptron) NeuralNetwork.createFromFile(Utils.TRAINED_NETWORK_FOLDER + name);
        this.name = name;

        // Load datasets
        learningDateSet = DataSet.load(Utils.DATASETS_FOLDER + name + "-learn");
        testDataSet = DataSet.load(Utils.DATASETS_FOLDER + name + "-test");
    }

    private void CreateDataSets(String name) throws IOException {
        Expression expression = new Expression(name, true);
        DataSet dataSet = new DataSet(Utils.NUM_INPUT_NODES, 1);
        for(int i = 0; i < expression.size(); i++)
            dataSet.addRow(new DataSetRow(expression.getFrames().get(i), expression.getResults().get(i)));

        SubSampling subSampling = new SubSampling(Utils.PERCENTAGE_TO_TRAIN, 100 - Utils.PERCENTAGE_TO_TRAIN);
        List<DataSet> dataSets = subSampling.sample(dataSet);
        learningDateSet = dataSets.get(0);
        testDataSet = dataSets.get(1);
    }

    /**
     * Sets the back propagation algorithm rules.
     * @param maxIterations
     * @param maxError
     * @param learningRate
     */
    private void SetBackPropagationRules(int maxIterations, double maxError, float learningRate) {
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.maxError = maxError;
    }

    /**
     * Makes the neural network learn the data set.
     * @param displayResults
     */
    public void LearnDataSet(boolean displayResults) {
        if(displayResults)
            System.out.println("\n\tNeural network '" + name + "' started learning.");

        Thread t = new Thread() {
            public void run() {
                long timeStart = System.currentTimeMillis();
                perceptron.learn(learningDateSet, backPropagation);
                learnTime = (System.currentTimeMillis() - timeStart) / 1000f;
            }
        };

        t.start();
        int prevIteration = 1;
        int curIteration;
        double error;
        ArrayList<Double> iterationErrors = new ArrayList<>();
        while(t.isAlive()) {
            curIteration = backPropagation.getCurrentIteration();
            if (curIteration > prevIteration) {
                error = backPropagation.getPreviousEpochError();
                iterationErrors.add(error);
                if(displayResults)
                    System.out.format("Iteration: %" + (maxIterations + "").length() + "d of %d - Error: %f of %f\n",
                        curIteration-1, maxIterations, error, maxError);
                prevIteration = curIteration;
            }
        }

        networkError = backPropagation.getTotalNetworkError();
        iterationErrors.add(networkError);

        if(displayResults) {
            System.out.format("Iteration: %" + (maxIterations + "").length() + "d of %d - Error: %f of %f\n",
                    maxIterations, maxIterations, networkError, maxError);
            System.out.println("\tNeural network '" + name + "' has finished learning. Total network error: " + networkError + "\n");
        }

        try{
            PrintWriter writer = new PrintWriter(Utils.LEARNING_ERROR_FOLDER + this.getName() + ".csv", "UTF-8");
                writer.println("Iterations, Error");
            for (int i = 0; i < iterationErrors.size(); i++)
                writer.format(Locale.GERMANY,"%d, \"%f\"\n",i, iterationErrors.get(i));
            writer.close();
        } catch (IOException e) {
            System.err.println("Couldn't write iteration errors to file\n" + e.getLocalizedMessage());
        }
    }

    /**
     * Tests the neural network using the test data set and saves the information to a file.
     * @param displayResults
     * @throws IOException
     */
    public double TestNeuralNetwork(boolean displayResults, DataSet dataSet) throws IOException {
        if(displayResults)
            System.out.println("\n\tNeural network started calculating.");

        ArrayList<Double> diffArray = new ArrayList<>();
        DataSet tmpDataSet = null;
        if(dataSet != null) {
            tmpDataSet = testDataSet;
            testDataSet = dataSet;
        }

        for(DataSetRow row : testDataSet.getRows()) {
            perceptron.setInput(row.getInput());
            perceptron.calculate();

            double desiredOutput = Double.parseDouble(Arrays.toString(row.getDesiredOutput()).replace("[", "").replace("]", ""));
            double actualOutput = Double.parseDouble(Arrays.toString(perceptron.getOutput()).replace("[", "").replace("]", ""));
            double diff = desiredOutput - actualOutput;
            double formulaDiff = Math.pow(diff, 2);
            diffArray.add(formulaDiff);

            if(displayResults)
                System.out.println("\tDesired output: " + desiredOutput + " \t|\t Actual output: " + actualOutput + " \t|\t Difference: " + diff);
        }

        double diffSum = 0;
        for(int i = 0; i < diffArray.size(); i++)
            diffSum += diffArray.get(i);
        double meanError = diffSum / diffArray.size();

        if(dataSet != null)
            testDataSet = tmpDataSet;
        if(displayResults)
            System.out.println("\n\tNumber of tests: " + diffArray.size() + ", average error: " + meanError + "\n");

        return meanError;
    }

    public Double TestNeuralNetwork(Expression expression) {
        perceptron.setInput(new DataSetRow(expression.getCoords()).getInput());
        perceptron.calculate();
        return Double.parseDouble(Arrays.toString(perceptron.getOutput()).replace("[", "").replace("]", ""));
    }

    public double TestNeuralNetwork(DataSet dataSet) {
        ArrayList<Double> diffArray = new ArrayList<>();

        for(DataSetRow row : dataSet.getRows()) {
            perceptron.setInput(row.getInput());
            perceptron.calculate();
            double output = Double.parseDouble(Arrays.toString(perceptron.getOutput()).replace("[", "").replace("]", ""));
            diffArray.add(output);
        }

        double diffSum = 0;
        for(int i = 0; i < diffArray.size(); i++)
            diffSum += diffArray.get(i);

        return diffSum / diffArray.size();
    }

    /**
     * Saves the neural network as a neural network file.
     * @param filename
     */
    public void SaveNeuralNetwork(String filename) {
        if(name == null)
            name = filename;
        perceptron.save(Utils.TRAINED_NETWORK_FOLDER + filename);
        learningDateSet.save(Utils.DATASETS_FOLDER + name + "-learn");
        testDataSet.save(Utils.DATASETS_FOLDER + name + "-test");
    }

    /**
     * Writes all the information to the corresponding file.
     * @throws IOException
     */
    public void SaveResultToFile() throws IOException {
        // If the network was loaded, there's no need to write to the file, since it's already been tested.
        if(isLoaded)
            return;

        // Creates file if it doesn't exists.
        File file = new File(Utils.PERFORMANCE_FOLDER + name + ".txt");
        file.createNewFile();

        String str = maxIterations + " " + maxError + " " + learningRate + " " + networkError + " " + learnTime;
        for(int i = 1; i < perceptron.getLayersCount()-1; i++)
            str += " " + (perceptron.getLayers().get(i).getNeuronsCount()-1);

        FileReader fr = new FileReader(Utils.PERFORMANCE_FOLDER + name + ".txt");
        BufferedReader br = new BufferedReader(fr);
        String line;
        int i = 0;

        boolean replace = false;
        // Detects where to put the current back propagation rules in the file, since the top most rules are the best for this network.
        while((line = br.readLine()) != null) {
            String[] msgSplit = line.split(" ");
            //int fileHiddenNodesAmount = Integer.parseInt(msgSplit[0]);
            int fileMaxIterations = Integer.parseInt(msgSplit[0]);
            double fileMaxError = Double.parseDouble(msgSplit[1]);
            float fileLearningRate = Float.parseFloat(msgSplit[2]);
            double fileNetworkError = Double.parseDouble(msgSplit[3]);

            ArrayList<Integer> fileHiddenLayersNodes = new ArrayList<>();
            for(int j = 5; j < msgSplit.length; j++) {
                int hiddenLayerNodes = Integer.parseInt(msgSplit[j]);
                fileHiddenLayersNodes.add(hiddenLayerNodes);
            }

            // If the rules are the same but the network error is higher or equal, there's no need to write to the file.
            if(fileHiddenLayersNodes.size() == (perceptron.getLayersCount()-2) &&
                    fileMaxIterations == maxIterations &&
                    fileMaxError == maxError &&
                    fileLearningRate == learningRate) {
                if(networkError < fileNetworkError) {
                    replace = true;
                    break;
                } else
                    return;
            }

            if(networkError < fileNetworkError)
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

    /**
     * Name getter.
     * @return
     */
    public String getName() {
        return name;
    }

    @Override
    public boolean equals(Object obj) {
        if(obj instanceof _NeuralNetwork) {
            _NeuralNetwork nn = (_NeuralNetwork) obj;
            return (nn.getName().equals(name));
        }
        return false;
    }
}