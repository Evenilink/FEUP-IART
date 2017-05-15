import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Expression {

    private ArrayList<String> frames;
    private ArrayList<String> results;
    private ArrayList<Double> xCoords;
    private ArrayList<Double> yCoords;
    private double xMax;
    private double yMax;
    private double xMin;
    private double yMin;

    private int size;

    public Expression(String datasetName) throws IOException {
        frames = new ArrayList<>();
        results = new ArrayList<>();
        xCoords = new ArrayList<>();
        yCoords = new ArrayList<>();

        this.ParseDatapoints(Utils.EXPRESSION_FOLDER + datasetName + "_datapoints.txt");
        this.ParseTargets(Paths.get(Utils.EXPRESSION_FOLDER + datasetName + "_targets.txt"));
        this.size = Math.min(frames.size(), results.size());
    }

    private void ParseDatapoints(String filepath) throws IOException {
        FileReader fr = new FileReader(filepath);
        BufferedReader br = new BufferedReader(fr);
        String line;

        while(true) {
            if ((line = br.readLine()) == null)
                return;

            if(!line.startsWith("0.0") && line.contains(" ")) {
                String[] lineSplit = line.split(" ");
                // i = 0 -> timestamp, which we ignore.
                for(int i = 1; i < lineSplit.length; i += 3) {
                    double x = Float.parseFloat(lineSplit[i]);
                    double y = Float.parseFloat(lineSplit[i+1]);
                    xCoords.add(x);
                    yCoords.add(y);
                }

                double max = 0, min = Double.MAX_VALUE;
                for(int i = 0; i < xCoords.size(); i++) {
                    if(xCoords.get(i) > max)
                        max = xCoords.get(i);
                    if(xCoords.get(i) < min)
                        min = xCoords.get(i);
                }

                xMax = max;
                xMin = min;

                max = 0;
                min = Double.MAX_VALUE;
                for(int i = 0; i < yCoords.size(); i++) {
                    if(yCoords.get(i) > max)
                        max = yCoords.get(i);
                    if(yCoords.get(i) < min)
                        min = yCoords.get(i);
                }

                yMax = max;
                yMin = min;

                String formatted = "";
                for(int i = 0; i < xCoords.size(); i++) {
                    formatted += (xCoords.get(i) - xMin) / (xMax - xMin) + " " + (yCoords.get(i) - yMin) / (yMax - yMin);
                    if(i != xCoords.size() - 1)
                        formatted += " ";
                }

                frames.add(formatted);
                xCoords.clear();
                yCoords.clear();
            }
        }
    }

    private void ParseTargets(Path fileLoc) {
        try {
            Files.lines(fileLoc).forEach(s -> results.add(s));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public ArrayList<String> getFrames() {
        return frames;
    }

    public ArrayList<String> getResults() {
        return results;
    }

    public int size() {
        return size;
    }
}