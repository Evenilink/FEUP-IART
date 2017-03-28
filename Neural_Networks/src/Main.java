public class Main {

    public static void main(String[] args) {
        if(args.length != 3) {
            System.out.println("Usage: java -jar Main <max_iterations> <max_error> <learning_rate> <filename>");
            return;
        }

        _NeuralNetwork neuralNetwork = new _NeuralNetwork(Integer.parseInt(args[0]), Double.parseDouble(args[1]), Float.parseFloat(args[2]));
    }

    public static void parser() {
        String filename = "a_affirmative_datapoints";


        Path fileLoc = Paths.get("../grammatical_facial_expression/" + filename + ".txt");
        ArrayList<Frame> expression = new ArrayList<>();

        try {
            Files.lines(fileLoc).forEach(s -> {
                // Discards first line
                if (!s.startsWith("0.0")) {

                    String value = s.substring(0, s.indexOf(" "));
                    s = s.substring(s.indexOf(" ") + 1);
                    double time = Double.parseDouble(value);
                    Frame frame = new Frame(time);

                    while(s.contains(" ")) {
                        value = s.substring(0, s.indexOf(" "));
                        s = s.substring(s.indexOf(" ") + 1);
                        frame.addToX(Double.parseDouble(value));

                        value = s.substring(0, s.indexOf(" "));
                        s = s.substring(s.indexOf(" ") + 1);
                        frame.addToY(Double.parseDouble(value));

                        if (s.contains(" ")) {
                            value = s.substring(0, s.indexOf(" "));
                            s = s.substring(s.indexOf(" ") + 1);
                        }
                        else
                            value = s;
                        frame.addToZ(Double.parseDouble(value));

                    }
                    expression.add(frame);
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
