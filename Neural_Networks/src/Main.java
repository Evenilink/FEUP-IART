import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class Main {

    private static Scanner scanner;
    private static int userInput;
    private static ArrayList<_NeuralNetwork> neuralNetworks;

    public static void main(String[] args) throws IOException {
        scanner = new Scanner(System.in);
        neuralNetworks = new ArrayList<>();
        Menu();
    }

    private static void Menu() throws IOException {
        System.out.println("NEURAL NETWORKS\n");

        do {
            System.out.print("1 - Create and make a Network learn\n2 - Test Network\n0 - Exit\nPlease select a sub-menu: ");
            userInput = scanner.nextInt();
            scanner.nextLine();     // Get's rid of the newline.

            switch (userInput) {
                case 1:
                    LearnNetwork();
                    break;
                case 2:
                    TestNetwork();
                    break;
                default: break;
            }
        } while(userInput != 0);

        System.out.println("Goodbye user.");
    }

    private static void LearnNetwork() throws IOException {
        System.out.print("\n\tEnter file with data examples for the network to learn: ");
        String filename = scanner.nextLine();

        Expression expression = new Expression(filename);
        DataSet learnDataSet = new DataSet(expression.GetFramesCount(), 1);

        int trainFramesAmount = Math.round(expression.getFrames().size() * Utils.PERCENTAGE_TO_TRAIN);
        for(int i = 0; i < trainFramesAmount; i++)
            learnDataSet.addRow(new DataSetRow(expression.getFrames().get(i), expression.getResults().get(i)));

        _NeuralNetwork neuralNetwork = new _NeuralNetwork(expression.GetFramesCount());
        neuralNetwork.LearnDataSet(learnDataSet);
        neuralNetwork.SaveNeuralNetwork(filename);

        neuralNetworks.add(neuralNetwork);
    }

    private static void TestNetwork() throws IOException {
        for(int i = 0; i < neuralNetworks.size(); i++)
            System.out.println("\t" + neuralNetworks.get(i));

        System.out.print("\n\tSelect a network to test");
        int networkToTest = scanner.nextInt();
        scanner.nextLine();

        System.out.print("\tEnter file with data examples for the network to test: ");
        String filename = scanner.nextLine();

        Expression expression = new Expression(filename);
        DataSet testDataSet = new DataSet(expression.GetFramesCount(), 1);
        int trainFramesAmount = Math.round(expression.getFrames().size() * Utils.PERCENTAGE_TO_TRAIN);

        for(int i = trainFramesAmount; i < expression.size(); i++)
            testDataSet.addRow(new DataSetRow(expression.getFrames().get(i), expression.getResults().get(i)));

        neuralNetworks.get(networkToTest).TestNeuralNetwork(testDataSet);

    }
}