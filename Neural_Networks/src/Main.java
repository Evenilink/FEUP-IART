import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        if(args.length != 1) {
            System.out.println("Usage: java -jar Main <filename>");
            return;
        }
        String filename = args[0];

        Expression expression = new Expression(filename);
        DataSet learnDataSet = new DataSet(expression.GetFramesCount(), 1);

        int trainFramesAmount = Math.round(expression.getFrames().size() * Utils.PERCENTAGE_TO_TRAIN);
        for(int i = 0; i < trainFramesAmount; i++)
            learnDataSet.addRow(new DataSetRow(expression.getFrames().get(i), expression.getResults().get(i)));

        DataSet testDataSet = new DataSet(expression.GetFramesCount(), 1);
        for(int i = trainFramesAmount; i < expression.size(); i++)
            testDataSet.addRow(new DataSetRow(expression.getFrames().get(i), expression.getResults().get(i)));

        _NeuralNetwork neuralNetwork = new _NeuralNetwork(expression.GetFramesCount());
        neuralNetwork.LearnDataSet(learnDataSet);
        neuralNetwork.TestNeuralNetwork(testDataSet);
        neuralNetwork.SaveNeuralNetwork(filename);
    }
}