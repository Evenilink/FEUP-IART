import org.neuroph.core.NeuralNetwork;

import java.io.*;
import java.util.*;

public class Main {

    private static Scanner scanner;
    private static int userInput;
    private static HashMap<String, _NeuralNetwork> neuralNetworks;

    public static void main(String[] args) throws IOException {
        scanner = new Scanner(System.in);
        scanner.useLocale(Locale.US);       // Forces '.' to be the decimal delimiter.
        neuralNetworks = new HashMap<>();
        LoadNetworks();
        Menu();
    }

    /**
     * Loads the created networks to the hash map, so that they can be tested without having to create them again.
     * @throws IOException
     */
    private static void LoadNetworks() throws IOException {
        System.out.println("Loading networks...");

        File folder = new File(Utils.TRAINED_NETWORK_FOLDER);
        File[] files = folder.listFiles();

        for(int i = 0; i < files.length; i++) {
            String fileName = files[i].getName();
            _NeuralNetwork neuralNetwork = new _NeuralNetwork(fileName);
            neuralNetworks.put(fileName, neuralNetwork);
        }
        System.out.println("Networks Loaded!\n");
    }

    /**
     * Friendly user interface.
     * @throws IOException
     */
    private static void Menu() throws IOException {
        System.out.println("*** NEURAL NETWORKS ***\n");

        do {
            System.out.print("1 - Create and make a Network learn\n2 - Test Network\n3 - Test an example\n4 - Apply the best rules to a network\n5 - Brute force a neural network to find the best performance\n0 - Exit\n\nPlease select a sub-menu: ");
            userInput = scanner.nextInt();
            scanner.nextLine();     // Get's rid of the newline.

            switch (userInput) {
                case 1: LearnNetwork(); break;
                case 2: TestNetwork(); break;
                case 3: TestExample(); break;
                case 4: ApplyBestRules(); break;
                case 5: BruteForce(); break;
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
        String filename;
        while(true) {
            System.out.print("\n\tWrite an expression name with data examples for the network to learn: ");
            filename = scanner.nextLine();

            File f = new File(Utils.EXPRESSION_FOLDER + "a_" + filename + "_datapoints.txt");
            if(!f.exists())
                System.out.println("\tThere are no examples available for '" + filename + "' expression.");
            else break;
        }

        System.out.print("\tMaximum iterations: ");
        int maxIterations = scanner.nextInt();
        System.out.print("\tMaximum error: ");
        double maxError = scanner.nextDouble();
        System.out.print("\tLearning rate: ");
        float learningRate = scanner.nextFloat();
        System.out.print("\tNumber of hidden nodes: ");
        int numHiddenNodes = scanner.nextInt();

        CreateNeuralNetwork(filename, numHiddenNodes, maxIterations, maxError, learningRate, true);
    }

    /**
     * Tests a networks and outputs the results to the console.
     * @throws IOException
     */
    private static void TestNetwork() throws IOException {
        String selectedNetworkName = ListLoadedNetworks();
        if(selectedNetworkName == null)
            return;

        neuralNetworks.get(selectedNetworkName).TestNeuralNetwork(true);
    }

    private static void TestExample() throws IOException {
        System.out.print("\n\tCopy your example here: ");
        String inputTest = scanner.nextLine();
        System.out.println();

        Expression expression = new Expression(inputTest, false);
        HashMap<String, Double> networksPerformance = new HashMap<>();
        Iterator<Map.Entry<String, _NeuralNetwork>> it = neuralNetworks.entrySet().iterator();
        while(it.hasNext()) {
            @SuppressWarnings("unchecked")
            Map.Entry<String, _NeuralNetwork> entry = it.next();
            _NeuralNetwork neuralNetwork = entry.getValue();

            Double performance = neuralNetwork.TestNeuralNetwork(expression);
            networksPerformance.put(neuralNetwork.getName(), performance);
        }

        double maxPerformance = 0;
        String facialExpression = null;
        Iterator<Map.Entry<String, Double>> itr = networksPerformance.entrySet().iterator();
        while(itr.hasNext()) {
            @SuppressWarnings("unchecked")
            Map.Entry<String, Double> entry = itr.next();

            Double performance = entry.getValue();
            String networkName = entry.getKey();
            if(performance > maxPerformance) {
                maxPerformance = performance;
                facialExpression = networkName;
            }

            System.out.println("\t" + networkName + ": " + performance);
        }

        System.out.println("\tThe entered input belongs to the facial expression '" + facialExpression + "' with output '" + maxPerformance + "'.\n");
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

        // Searches the best learning rate.
        for(float learningRate = 0.0f; learningRate <= 0.95f; learningRate += Utils.LEARNING_RATE_INCREMENT) {
            CreateNeuralNetwork(selectedNetworkName, Utils.DEFAULT_NUM_HIDDEN_LAYERS, Utils.DEFAULT_MAX_ITERATIONS, Utils.DEFAULT_MAX_ERROR, learningRate, false);
            System.out.println("\t\tCreated network with learning rate = '" + learningRate + "'.");
        }

        String bestRules = GetBestRules(selectedNetworkName);
        String[] msgSplit = bestRules.split(" ");
        float learningRate = Float.parseFloat(msgSplit[2]);

        // Searches the best number of hidden nodes.
        // Formula: number of input nodes * number of hidden nodes < number of examples
        int numHiddenNodes = 2;
        while(Utils.NUM_INPUT_NODES * numHiddenNodes < neuralNetworks.get(selectedNetworkName).getLearningDateSet().getRows().size()) {
            CreateNeuralNetwork(selectedNetworkName, numHiddenNodes, Utils.DEFAULT_MAX_ITERATIONS, Utils.DEFAULT_MAX_ERROR, learningRate, false);
            System.out.println("\t\tCreated network with '" + numHiddenNodes + "' hidden nodes.");
            numHiddenNodes++;
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

        return msgSplit[0] + " " + msgSplit[1] + " " + msgSplit[2] + " " + msgSplit[3];
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
        _NeuralNetwork neuralNetwork = new _NeuralNetwork(networkName, Utils.NUM_INPUT_NODES, hiddenNodesAmount, maxIterations, maxError, learningRate);
        neuralNetwork.LearnDataSet(displayResults);
        neuralNetwork.SaveNeuralNetwork(networkName);
        neuralNetwork.TestNeuralNetwork(false);

        // If a previous neural network of the same name exists, it's replaced by the just created one.
        if(neuralNetworks.containsKey(neuralNetwork.getName()))
            neuralNetworks.remove(neuralNetwork.getName());
        neuralNetworks.put(neuralNetwork.getName(), neuralNetwork);
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
            i++;
        }

        System.out.print("\n\tSelect a network to test: ");
        int networkToTest = scanner.nextInt(); scanner.nextLine();
        return neuralNetworkNames.get(networkToTest-1);
    }
}