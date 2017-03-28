import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

public class Main {

    public static void main(String[] args) {
        if(args.length != 4) {
            System.out.println("Usage: java -jar Main <max_iterations> <max_error> <learning_rate> <filename>");
            return;
        }

        Expression expression = new Expression("a_affirmative");

        DataSet dataSet = new DataSet(300, 1);
        for(int i = 0; i < expression.size(); i++)
            dataSet.addRow(new DataSetRow(expression.getFrames().get(i), expression.getResults().get(i)));

        _NeuralNetwork neuralNetwork = new _NeuralNetwork(Integer.parseInt(args[0]), Double.parseDouble(args[1]), Float.parseFloat(args[2]));
        neuralNetwork.learnDataSet(dataSet);
        neuralNetwork.testNeuralNetwork(dataSet);
        neuralNetwork.saveNeuralNetwork(args[3]);
    }
}
