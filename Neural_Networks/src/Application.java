import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

import java.io.*;
import java.util.*;

public class Application {

    private static Scanner scanner;
    private static int userInput;
    private static HashMap<String, _NeuralNetwork> neuralNetworks;

    public static void main(String[] args) throws IOException {
        // Create folders if they don't exist
        File folder = new File(Utils.EXPRESSION_FOLDER);
        if (!folder.exists()) folder.mkdirs();
        folder = new File(Utils.TRAINED_NETWORK_FOLDER);
        if (!folder.exists()) folder.mkdirs();
        folder = new File(Utils.DATASETS_FOLDER);
        if (!folder.exists()) folder.mkdirs();
        folder = new File(Utils.PERFORMANCE_FOLDER);
        if (!folder.exists()) folder.mkdirs();
        folder = new File(Utils.LEARNING_ERROR_FOLDER);
        if (!folder.exists()) folder.mkdirs();

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
        File folder = new File(Utils.TRAINED_NETWORK_FOLDER);
        File[] files = folder.listFiles();

        if(files != null && files.length > 0) {
            System.out.println("Loading networks...");
            for(int i = 0; i < files.length; i++) {
                String fileName = files[i].getName();
                _NeuralNetwork neuralNetwork = new _NeuralNetwork(fileName);
                neuralNetworks.put(fileName, neuralNetwork);
            }
            System.out.println("Networks Loaded!\n");
        }
    }

    /**
     * Friendly user interface.
     * @throws IOException
     */
    private static void Menu() throws IOException {
        System.out.println("*** NEURAL NETWORKS ***\n");

        do {
            System.out.print("1 - Create and make a Network learn\n2 - Create all networks\n3 - Test Network\n4 - Test an example\n5 - Test a file\n6 - Apply the best rules to a network\n0 - Exit\n\nPlease select a sub-menu: ");
            userInput = scanner.nextInt();
            scanner.nextLine();     // Get's rid of the newline.

            switch (userInput) {
                case 1: LearnNetwork(false); break;
                case 2: LearnNetwork(true); break;
                case 3: TestNetwork(); break;
                case 4: TestExample(); break;
                case 5: TestFile(); break;
                case 6: ApplyBestRules(); break;
                default: break;
            }
        } while(userInput != 0);

        System.out.println("Goodbye user.");
    }

    /**
     * Creates a new network and makes it learn.
     * @throws IOException
     */
    private static void LearnNetwork(boolean allNetworks) throws IOException {
        String filename = null;

        if(!allNetworks) {
            while (true) {
                System.out.print("\n\tWrite an expression name with data examples for the network to learn: ");
                filename = scanner.nextLine();

                File f = new File(Utils.EXPRESSION_FOLDER + "a_" + filename + "_datapoints.txt");
                File f1 = new File(Utils.EXPRESSION_FOLDER + filename + "_datapoints.txt");
                if (!f.exists() && !f1.exists())
                    System.out.println("\tThere are no examples available for '" + filename + "' expression.");
                else break;
            }
        }

        System.out.print("\tMaximum iterations: ");
        int maxIterations = scanner.nextInt();
        System.out.print("\tMaximum error: ");
        double maxError = scanner.nextDouble();
        System.out.print("\tLearning rate: ");
        float learningRate = scanner.nextFloat();

        int numHiddenLayers = -1;
        while(numHiddenLayers < 1) {
            System.out.print("\tNumber of hidden layers: ");
            numHiddenLayers = scanner.nextInt();
        }

        ArrayList<Integer> numHiddenNodes = new ArrayList<>();
        for(int i = 0; i < numHiddenLayers; i++) {
            int hiddenNodes = -1;

            while(hiddenNodes < 1) {
                System.out.print("\tNumber of hidden nodes in hidden layer " + (i+1) + ": ");
                hiddenNodes = scanner.nextInt();
            }

            numHiddenNodes.add(hiddenNodes);
        }

        if(allNetworks) {
            File folder = new File(Utils.EXPRESSION_FOLDER);
            File[] files = folder.listFiles();

            if(files != null && files.length > 0) {
                int i = 0;
                while(true) {
                    filename = files[i].getName();
                    String[] expressionFile = filename.split("_");
                    if(expressionFile[0].equals("b"))
                        break;

                    String expression = "";
                    for(int j = 1; j < expressionFile.length; j++) {
                        if(expressionFile[j].equals("datapoints.txt"))
                            break;
                        expression += expressionFile[j];
                        if(j+2 != expressionFile.length)
                            expression += "_";
                    }
                    CreateNeuralNetwork(expression, numHiddenNodes, maxIterations, maxError, learningRate, true);
                    i+=2;
                }
            }
        } else
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

        _NeuralNetwork neuralNetwork = neuralNetworks.get(selectedNetworkName);
        neuralNetwork.TestNeuralNetwork(true, null);
        neuralNetwork.SaveResultToFile();
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

            Double output = neuralNetwork.TestNeuralNetwork(expression);
            networksPerformance.put(neuralNetwork.getName(), output);
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

    private static void TestFile() throws IOException {
        System.out.print("\n\tSelect a file: ");
        String filePath = scanner.nextLine();

        Expression expression = new Expression(filePath);
        DataSet dataSet = new DataSet(Utils.NUM_INPUT_NODES, 1);
        for(int i = 0; i < expression.size(); i++)
            dataSet.addRow(new DataSetRow(expression.getFrames().get(i), "-1"));

        HashMap<String, Double> networksPerformance = new HashMap<>();
        Iterator<Map.Entry<String, _NeuralNetwork>> it = neuralNetworks.entrySet().iterator();
        while(it.hasNext()) {
            @SuppressWarnings("unchecked")
            Map.Entry<String, _NeuralNetwork> entry = it.next();
            _NeuralNetwork neuralNetwork = entry.getValue();

            Double output = neuralNetwork.TestNeuralNetwork(dataSet);
            networksPerformance.put(neuralNetwork.getName(), output);
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

        System.out.println("\tThe entered file has frames belonging to the facial expression '" + facialExpression + "' with output '" + maxPerformance + "'.\n");
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

        int maxIterations = Integer.parseInt(msgSplit[0]);
        double maxError = Double.parseDouble(msgSplit[1]);
        float learningRate = Float.parseFloat(msgSplit[2]);

        ArrayList<Integer> hiddenLayersNodes = new ArrayList<>();
        for(int i = 3; i < msgSplit.length; i++) {
            int hiddenLayerNodes = Integer.parseInt(msgSplit[i]);
            hiddenLayersNodes.add(hiddenLayerNodes);
        }

        System.out.println("\tApplying best rules:\n\t\tNumber of hidden layers: " + hiddenLayersNodes.size() + "\n\t\tMaximum iterations: " + maxIterations + "\n\t\tMaximum error: " + maxError + "\n\t\tLearning rate: " + learningRate + "\n");
        CreateNeuralNetwork(selectedNetworkName, hiddenLayersNodes, maxIterations, maxError, learningRate, true);
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

        String msg = msgSplit[0] + " " + msgSplit[1] + " " + msgSplit[2];
        for(int i = 5; i < msgSplit.length; i++)
            msg += " " + msgSplit[i];

        return msg;
    }

    /**
     * Creates a new neural network.
     * @param networkName
     * @param maxIterations
     * @param maxError
     * @param learningRate
     * @throws IOException
     */
    private static void CreateNeuralNetwork(String networkName, ArrayList<Integer> hiddenNodesLayerAmount, int maxIterations, double maxError, float learningRate, boolean displayResults) throws IOException {
        _NeuralNetwork neuralNetwork = new _NeuralNetwork(networkName, hiddenNodesLayerAmount, maxIterations, maxError, learningRate);
        neuralNetwork.LearnDataSet(displayResults);
        neuralNetwork.SaveNeuralNetwork(networkName);
        neuralNetwork.SaveResultToFile();

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