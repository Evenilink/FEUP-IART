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

    private static void LearnNetwork() throws IOException {
        System.out.print("\n\tEnter file with data examples for the network to learn: ");
        String filename = scanner.nextLine();

        System.out.print("\tMaximum iterations: ");
        int maxIterations = scanner.nextInt();

        System.out.print("\tMaximum error: ");
        double maxError = 0.001;//scanner.nextDouble();

        System.out.print("\tLearning rate: ");
        float learningRate = 0.01f;//scanner.nextFloat();

        CreateNeuralNetwork(filename, maxIterations, maxError, learningRate);
    }

    private static void TestNetwork() throws IOException {
        String selectedNetworkName = ListLoadedNetworks();
        if(selectedNetworkName == null)
            return;

        Expression expression = new Expression(selectedNetworkName);
        DataSet testDataSet = new DataSet(expression.GetFramesCount(), 1);
        int trainFramesAmount = Math.round(expression.getFrames().size() * Utils.PERCENTAGE_TO_TRAIN);

        for(int i = trainFramesAmount; i < expression.size(); i++)
            testDataSet.addRow(new DataSetRow(expression.getFrames().get(i), expression.getResults().get(i)));

        neuralNetworks.get(selectedNetworkName).TestNeuralNetwork(testDataSet);
    }

    private static void ApplyBestRules() throws IOException {
        String selectedNetworkName = ListLoadedNetworks();
        if(selectedNetworkName == null)
            return;

        FileReader fr = new FileReader(Utils.PERFORMANCE_FOLDER + selectedNetworkName + ".txt");
        BufferedReader br = new BufferedReader(fr);
        String line;

        line = br.readLine();
        String[] msgSplit = line.split(" ");

        int maxIterations = Integer.parseInt(msgSplit[0]);
        double maxError = Double.parseDouble(msgSplit[1]);
        float learningRate = Float.parseFloat(msgSplit[2]);

        br.close();
        fr.close();

        System.out.println("\tApplying best rules:\n\t\tMaximum iterations: " + maxIterations + "\n\t\tMaximum error: " + maxError + "\n\t\tLearning rate: " + learningRate + "\n");
        CreateNeuralNetwork(selectedNetworkName, maxIterations, maxError, learningRate);
    }

    private static void BruteForce() {

    }

    /**
     * Creates a new neural network.
     * @param networkName
     * @param maxIterations
     * @param maxError
     * @param learningRate
     * @throws IOException
     */
    private static void CreateNeuralNetwork(String networkName, int maxIterations, double maxError, float learningRate) throws IOException {
        Expression expression = new Expression(networkName);
        DataSet learnDataSet = new DataSet(expression.GetFramesCount(), 1);

        int trainFramesAmount = Math.round(expression.getFrames().size() * Utils.PERCENTAGE_TO_TRAIN);
        for(int i = 0; i < trainFramesAmount; i++)
            learnDataSet.addRow(new DataSetRow(expression.getFrames().get(i), expression.getResults().get(i)));

        _NeuralNetwork neuralNetwork = new _NeuralNetwork(networkName, expression.GetFramesCount(), maxIterations, maxError, learningRate);
        neuralNetwork.LearnDataSet(learnDataSet);
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
}