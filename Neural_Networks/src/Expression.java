import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

public class Expression {

    private static final String expressionsFolder = "../grammatical_facial_expression/";
    private ArrayList<String> frames;
    private ArrayList<String> results;
    private int size;
    private int framesCount;

    private int hResolution = 640;
    private int vResolution = 480;
    private int zMaxDeepth = 4000;

    public Expression(String datasetName) {
        frames = new ArrayList<>();
        results = new ArrayList<>();

        this.ParseDatapoints(Paths.get(expressionsFolder + datasetName + "_datapoints.txt"));
        this.ParseTargets(Paths.get(expressionsFolder + datasetName + "_targets.txt"));
        this.size = Math.min(frames.size(), results.size());

        this.framesCount = 0;
        String s = frames.get(0);
        do {
            s = s.substring(s.indexOf(" ") + 1);
            this.framesCount++;
        } while(s.contains(" "));
        this.framesCount++;
    }

    private void ParseDatapoints(Path fileLoc) {
        try {
            Files.lines(fileLoc).forEach(s -> {
                String formatted = "";

                // Discards first line
                if (!s.startsWith("0.0")) {
                    if (s.contains(" ")) {
                        String[] lineSplit = s.split(" ");
                        // i = 0 -> timestamp, which we ignore.
                        for(int i = 1; i < lineSplit.length; i += 3) {
                            double x = Float.parseFloat(lineSplit[i]) / hResolution;
                            double y = Float.parseFloat(lineSplit[i+1]) / vResolution;
                            double z = Float.parseFloat(lineSplit[i+2]) / zMaxDeepth;
                            formatted += x + " " + y + " " + z;
                            if(i != lineSplit.length - 3)
                                formatted += " ";
                        }
                    }
                    frames.add(formatted);
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
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

    public int GetFramesCount() {
        return framesCount;
    }
}