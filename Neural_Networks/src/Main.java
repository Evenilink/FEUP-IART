import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

public class Main {

    public static void main(String[] args) {
        if(args.length != 4) {
            System.out.println("Usage: java -jar Main <max_iterations> <max_error> <learning_rate> <filename>");
            return;
        }

        int maxIterations = Integer.parseInt(args[0]);
        double maxError = Double.parseDouble(args[1]);
        float learningRate = Float.parseFloat(args[2]);
        String filename = args[3];

        Expression expression = new Expression(filename);
        DataSet dataSet = new DataSet(expression.GetFramesCount(), 1);

        for(int i = 0; i < expression.size(); i++)
            dataSet.addRow(new DataSetRow(expression.getFrames().get(i), expression.getResults().get(i)));

        _NeuralNetwork neuralNetwork = new _NeuralNetwork(maxIterations, maxError, learningRate, expression.GetFramesCount());
        neuralNetwork.LearnDataSet(dataSet);
        neuralNetwork.TestNeuralNetwork(dataSet);
        neuralNetwork.SaveNeuralNetwork(args[3]);
    }
}
