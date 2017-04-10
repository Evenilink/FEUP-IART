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
        DataSet dataSet = new DataSet(expression.GetFramesCount(), 1);

        for(int i = 0; i < expression.size(); i++)
            dataSet.addRow(new DataSetRow(expression.getFrames().get(i), expression.getResults().get(i)));

        _NeuralNetwork neuralNetwork = new _NeuralNetwork(expression.GetFramesCount());
        neuralNetwork.LearnDataSet(dataSet);
        neuralNetwork.TestNeuralNetwork(dataSet);
        neuralNetwork.SaveNeuralNetwork(filename);
    }
}