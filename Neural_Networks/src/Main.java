import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

import java.io.*;
import java.util.*;

public class Main {

    private static Scanner scanner;
    private static int userInput;
    private static HashMap<String, _NeuralNetwork> neuralNetworks;

    public static void main(String[] args) throws IOException {
        scanner = new Scanner(System.in);
        neuralNetworks = new HashMap<>();
        LoadNetworks();
        Menu();
    }

    /**
     * Loads the created networks to the hash map, so that they can be tested without having to create them again.
     * @throws IOException
     */
    private static void LoadNetworks() throws IOException {
        File folder = new File(Utils.TRAINED_NETWORK_FOLDER);
        File[] files = folder.listFiles();

        for(int i = 0; i < files.length; i++) {
            String fileName = files[i].getName();
            _NeuralNetwork neuralNetwork = new _NeuralNetwork();
            neuralNetwork.LoadNeuralNetwork(fileName);
            neuralNetworks.put(fileName, neuralNetwork);
        }
    }

    /**
     * Friendly user interface.
     * @throws IOException
     */
    private static void Menu() throws IOException {
        System.out.println("*** NEURAL NETWORKS ***\n");

        do {
            System.out.print("1 - Create and make a Network learn\n2 - Test Network\n3 - Apply the best rules to a network\n4 - Brute force a neural network to find the best performance\n0 - Exit\n\nPlease select a sub-menu: ");
            userInput = scanner.nextInt();
            scanner.nextLine();     // Get's rid of the newline.

            switch (userInput) {
                case 1: LearnNetwork(); break;
                case 2: TestNetwork(); break;
                case 3: ApplyBestRules(); break;
                case 4: BruteForce(); break;
                default: break;
            }
        } while(userInput != 0);

        System.out.println("Goodbye user.");
    }

    /**
     * Creates a new network and makes it learn.
     * @throws IOException
     */
    private static void LearnNetwork() throws IOException {
        System.out.print("\n\tEnter file with data examples for the network to learn: ");
        String filename = scanner.nextLine();
        System.out.print("\tMaximum iterations: ");
        int maxIterations = scanner.nextInt();
        System.out.print("\tMaximum error: ");
        double maxError = 0.001;//scanner.nextDouble();
        System.out.print("\tLearning rate: ");
        float learningRate = 0.01f;//scanner.nextFloat();

        CreateNeuralNetwork(filename, Utils.NUM_HIDDEN_LAYERS, maxIterations, maxError, learningRate, true);
    }

    /**
     * Tests a networks and outputs the results to the console.
     * @throws IOException
     */
    private static void TestNetwork() throws IOException {
        String selectedNetworkName = ListLoadedNetworks();
        if(selectedNetworkName == null)
            return;

        TestNeuralNetwork(selectedNetworkName, true);
    }

    /**
     * Applies to the specified network the best possible rules in the corresponding file (which are displayed at the top of the file).
     * @throws IOException
     */
    private static void ApplyBestRules() throws IOException {
        String selectedNetworkName = ListLoadedNetworks();
        if(selectedNetworkName == null)
            return;

        String bestRules = GetBestRules(selectedNetworkName);
        String[] msgSplit = bestRules.split(" ");

        int hiddenNodesAmount = Integer.parseInt(msgSplit[0]);
        int maxIterations = Integer.parseInt(msgSplit[1]);
        double maxError = Double.parseDouble(msgSplit[2]);
        float learningRate = Float.parseFloat(msgSplit[3]);

        System.out.println("\tApplying best rules:\n\t\tMaximum iterations: " + maxIterations + "\n\t\tMaximum error: " + maxError + "\n\t\tLearning rate: " + learningRate + "\n");
        CreateNeuralNetwork(selectedNetworkName, hiddenNodesAmount, maxIterations, maxError, learningRate, true);
    }

    private static void BruteForce() throws IOException {
        String selectedNetworkName = ListLoadedNetworks();
        if(selectedNetworkName == null)
            return;

        for(float learningRate = Utils.LEARNING_RATE_INCREMENT; learningRate <= 0.9f; learningRate += 0.3f) {
            CreateNeuralNetwork(selectedNetworkName, Utils.NUM_HIDDEN_LAYERS, Utils.MAX_ITERATIONS, Utils.MAX_ERROR, learningRate, false);
            TestNeuralNetwork(selectedNetworkName, false);
            System.out.println("\t\tCreated network with learning rate = '" + learningRate + "'.");
        }

        String bestRules = GetBestRules(selectedNetworkName);
        String[] msgSplit = bestRules.split(" ");
        float learningRate = Float.parseFloat(msgSplit[2]);

        for(int numHiddenNodes = 5; numHiddenNodes <= 15; numHiddenNodes++) {
            CreateNeuralNetwork(selectedNetworkName, numHiddenNodes, Utils.MAX_ITERATIONS, Utils.MAX_ERROR, learningRate, false);
            TestNeuralNetwork(selectedNetworkName, false);
            System.out.println("\t\tCreated network with '" + numHiddenNodes + "' hidden nodes.");
        }
    }

    /**
     * Opens the performance file of the respective network and returns the best rules for it.
     * @param neuralNetworkName
     * @return
     * @throws IOException
     */
    private static String GetBestRules(String neuralNetworkName) throws IOException {
        FileReader fr = new FileReader(Utils.PERFORMANCE_FOLDER + neuralNetworkName + ".txt");
        BufferedReader br = new BufferedReader(fr);
        String line;

        line = br.readLine();
        String[] msgSplit = line.split(" ");

        br.close();
        fr.close();

        return msgSplit[0] + " " + msgSplit[1] + " " + msgSplit[2];
    }

    /**
     * Creates a new neural network.
     * @param networkName
     * @param maxIterations
     * @param maxError
     * @param learningRate
     * @throws IOException
     */
    private static void CreateNeuralNetwork(String networkName, int hiddenNodesAmount, int maxIterations, double maxError, float learningRate, boolean displayResults) throws IOException {
        Expression expression = new Expression(networkName);
        DataSet learnDataSet = new DataSet(expression.GetFramesCount(), 1);

        int trainFramesAmount = Math.round(expression.getFrames().size() * Utils.PERCENTAGE_TO_TRAIN);
        for(int i = 0; i < trainFramesAmount; i++)
            learnDataSet.addRow(new DataSetRow(expression.getFrames().get(i), expression.getResults().get(i)));

        _NeuralNetwork neuralNetwork = new _NeuralNetwork(networkName, expression.GetFramesCount(), hiddenNodesAmount, maxIterations, maxError, learningRate);
        neuralNetwork.LearnDataSet(learnDataSet, displayResults);
        neuralNetwork.SaveNeuralNetwork(networkName);

        // If a previous neural network of the same name exists, it's replaced by the just created one.
        if(neuralNetworks.containsKey(neuralNetwork.GetName()))
            neuralNetworks.remove(neuralNetwork.GetName());
        neuralNetworks.put(neuralNetwork.GetName(), neuralNetwork);
    }

    /**
     * Lists all the loaded networks and returns the name of the selected by the user.
     * @return
     */
    private static String ListLoadedNetworks() {
        if(neuralNetworks.size() == 0) {
            System.out.println("\n\tThere are no available networks.\n");
            return null;
        }

        System.out.println("\n\tAvailable networks:");

        int i = 1;
        ArrayList<String> neuralNetworkNames = new ArrayList<>();
        Iterator it = neuralNetworks.entrySet().iterator();
        while(it.hasNext()) {
            @SuppressWarnings("unchecked")
            Map.Entry<String, NeuralNetwork> entry = (Map.Entry<String, NeuralNetwork>) it.next();
            String name = entry.getKey();
            neuralNetworkNames.add(name);
            System.out.println("\t" + i + " - " + name);
        }

        System.out.print("\n\tSelect a network to test: ");
        int networkToTest = scanner.nextInt(); scanner.nextLine();
        return neuralNetworkNames.get(networkToTest-1);
    }

    /**
     * Tests a neural network.
     * @param neuralNetworkName
     * @param displayResults
     * @throws IOException
     */
    private static void TestNeuralNetwork(String neuralNetworkName, boolean displayResults) throws IOException {
        Expression expression = new Expression(neuralNetworkName);
        DataSet testDataSet = new DataSet(expression.GetFramesCount(), 1);
        int trainFramesAmount = Math.round(expression.getFrames().size() * Utils.PERCENTAGE_TO_TRAIN);

        for(int i = trainFramesAmount; i < expression.size(); i++)
            testDataSet.addRow(new DataSetRow(expression.getFrames().get(i), expression.getResults().get(i)));

        neuralNetworks.get(neuralNetworkName).TestNeuralNetwork(testDataSet, displayResults);
    }
}