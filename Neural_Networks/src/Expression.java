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

    public Expression(String datasetName) {
        frames = new ArrayList<>();
        results = new ArrayList<>();
        this.parse(Paths.get(expressionsFolder + datasetName + "_datapoints.txt"), frames);
        this.parse(Paths.get(expressionsFolder + datasetName + "_targets.txt"), results);
        this.size = Math.min(frames.size(), results.size());
        this.framesCount = 0;
        String s = frames.get(0);
        while(s.contains(" ")) {
            s = s.substring(s.indexOf(" ") + 1);
            this.framesCount++;
        }

    }

    private void parse(Path fileLoc, ArrayList<String> destination) {
        try {
            Files.lines(fileLoc).forEach(s -> {
                // Discards first line
                if (!s.startsWith("0.0")) {
                    if (s.contains(" "))
                        s = s.substring(s.indexOf(" ") + 1);
                    destination.add(s);
                }
            });
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

    public int getFramesCount() {
        return framesCount;
    }
}