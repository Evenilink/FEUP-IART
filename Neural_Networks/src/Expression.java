import java.io.BufferedReader;
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

    private ArrayList<Double> coords;

    public Expression(String datasetName, boolean isDatasetName) throws IOException {
        xCoords = new ArrayList<>();
        yCoords = new ArrayList<>();

        if (datasetName.startsWith("a_") || datasetName.startsWith("b_")) {
            this.ParseDatapoints(Utils.EXPRESSION_FOLDER + datasetName + "_datapoints.txt");
            this.ParseTargets(Paths.get(Utils.EXPRESSION_FOLDER + datasetName + "_targets.txt"));
        } else {
            this.ParseDatapoints(Utils.EXPRESSION_FOLDER + "a_" + datasetName + "_datapoints.txt");
            this.ParseDatapoints(Utils.EXPRESSION_FOLDER + "b_" + datasetName + "_datapoints.txt");
            this.ParseTargets(Paths.get(Utils.EXPRESSION_FOLDER + "a_" + datasetName + "_targets.txt"));
            this.ParseTargets(Paths.get(Utils.EXPRESSION_FOLDER + "b_" + datasetName + "_targets.txt"));
        }

        this.size = Math.min(frames.size(), results.size());

        if(isDatasetName) {
            frames = new ArrayList<>();
            results = new ArrayList<>();

            this.ParseDatapoints(Utils.EXPRESSION_FOLDER + datasetName + "_datapoints.txt");
            this.ParseTargets(Paths.get(Utils.EXPRESSION_FOLDER + datasetName + "_targets.txt"));
            this.size = Math.min(frames.size(), results.size());
        } else {
            coords = new ArrayList<>();
            ParseFrame(datasetName);
        }
    }

    private void ParseFrame(String frame) {
        String[] frameSplit = frame.split(" ");
        for(int i = 0; i < frameSplit.length; i += 3) {
            double x = Float.parseFloat(frameSplit[i]);
            double y = Float.parseFloat(frameSplit[i+1]);
            xCoords.add(x);
            yCoords.add(y);
        }

        BuildCoords();
    }

    private void BuildCoords() {
        double xMax = 0, xMin = Double.MAX_VALUE;
        double yMax = 0, yMin = Double.MAX_VALUE;

        for(int i = 0; i < xCoords.size(); i++) {
            if(xCoords.get(i) > xMax)
                xMax = xCoords.get(i);
            if(xCoords.get(i) < xMin)
                xMin = xCoords.get(i);
            if(yCoords.get(i) > yMax)
                yMax = yCoords.get(i);
            if(yCoords.get(i) < yMin)
                yMin = yCoords.get(i);
        }

        for(int i = 0; i < xCoords.size(); i++) {
            coords.add((xCoords.get(i) - xMin) / (xMax - xMin));
            coords.add((yCoords.get(i) - yMin) / (yMax - yMin));
        }

        xCoords.clear();
        yCoords.clear();
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

    public ArrayList<Double> getCoords() {
        return coords;
    }

    public int size() {
        return size;
    }
}