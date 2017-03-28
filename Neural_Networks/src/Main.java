public class Main {

    public static void main(String[] args) {
        if(args.length != 3) {
            System.out.println("Usage: java -jar Main <max_iterations> <max_error> <learning_rate> <filename>");
            return;
        }

        _NeuralNetwork neuralNetwork = new _NeuralNetwork(Integer.parseInt(args[0]), Double.parseDouble(args[1]), Float.parseFloat(args[2]));
    }
}
