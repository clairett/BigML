import java.io.*;
import java.util.*;

/**
 * Created by tiantian on 10/4/16.
 */


public class LR {
    HashMap<String, Integer> map = new HashMap<String, Integer>() {{
        put("Agent", 0);
        put("other", 1);
        put("Organisation", 2);
        put("TimePeriod", 3);
        put("Device", 4);
        put("Activity", 5);
        put("ChemicalSubstance", 6);
        put("MeanOfTransportation", 7);
        put("SportsSeason", 8);
        put("Biomolecule", 9);
        put("Work", 10);
        put("CelestialBody", 11);
        put("Event", 12);
        put("Person", 13);
        put("Species", 14);
        put("Place", 15);
        put("Location", 16);

    }};

    double overflow = 20;

    int N;
    int[][] A;
    double[][] B;
    int k =0;


    public LR(int N) throws IOException {
        this.N = N;
        this.A = new int[17][N];
        this.B = new double[17][N];
    }


    public static void main(String[] args) throws IOException {
        int N = Integer.parseInt(args[0]);
        double eta = Double.parseDouble(args[1]);
        double mu = Double.parseDouble(args[2]);
        int T = Integer.parseInt(args[3]);
        int trainingSize = Integer.parseInt(args[4]);
        BufferedReader br1 = new BufferedReader(new InputStreamReader(System.in));
        BufferedReader br2 = new BufferedReader(new FileReader(new File(args[5])));

        LR lr = new LR(N);

        for (int t = 1; t <= T; t++) {
            // for each t, compute A and B
            lr.trainModel(br1, eta/(t*t), mu,  t, trainingSize);
        }
        lr.testModel(br2);

//        lr.testModelLabel(br2);

    }

    public void testModel(BufferedReader br) throws IOException {
        String l = null;
        while ((l = br.readLine()) != null) {
            String[] line = l.split("\t");
            String labels = line[1];
            @SuppressWarnings("unchecked")
            HashSet<Integer> X = tokenizeDoc(line[2]);
            double[] p = sigmoid(X);

            String ret = "";
            for (int i = 0; i < p.length; i++) {
                if (p[i] > 0.5) {
                    if (i == p.length - 1)
                        ret += getKeyFromValue(map, i) + "\t" + p[i];
                    else
                        ret += getKeyFromValue(map, i) + "\t" + p[i] + ",";
                }
            }
            System.out.println(ret);

        }
    }

    public void testModelLabel(BufferedReader br) throws IOException {
        String l = null;
        while ((l = br.readLine()) != null) {
            String[] line = l.split("\t");
            String labels = line[1];

            System.out.println(labels);

        }
    }

    public void trainModel(BufferedReader br, double lambda, double mu, int t, int trainingSize) throws IOException {
        String l;
//        double LCL = 0;
        while ((l = br.readLine()) != null) {
            String[] line = l.split("\t");
            String[] labels = line[1].split(",");
            @SuppressWarnings("unchecked")
            HashSet<Integer> X = tokenizeDoc(line[2]); // create X vector
            double[] p = sigmoid(X);   // compute prob

//            for (double prob:p) {
//                LCL += Math.log(prob);
//            }

            k++;
            for (String label: map.keySet()) {
                int index = map.get(label);

                for (int j : X) {
                    B[index][j] *= Math.pow((1 - 2 * lambda * mu), k - A[index][j]);
                    if (labelExists(labels, label)) {
                        B[index][j] += lambda * (1 - p[index]);
                    } else {
                        B[index][j] += lambda * (0 - p[index]);
                    }
                    A[index][j] = k;
                }
            }

            if (k == trainingSize*t) break;
        }

        for (String label: map.keySet()) {
            int index = map.get(label);
            for (int i = 0; i < N; i++) {
                B[index][i] *= Math.pow((1 - 2 * lambda * mu), k - A[index][i]);
                A[index][i] = k;
            }
        }

        // after each iteration, compute LCL
//        System.out.println(t + "\t" + LCL);
    }

    public double[] sigmoid(HashSet<Integer> X) {
        double[] p = new double[17];

        for (int j = 0; j < 17; j++) {
            double sum = 0.0;
            for (int i:X) {
                sum += B[j][i];
            }
            if (sum > overflow) sum = overflow;
            else if (sum < -overflow) sum = -overflow;
            double exp = Math.exp(sum);
            p[j] = exp/(1+exp);
        }
        return p;
    }

    private HashSet<Integer> tokenizeDoc(String cur_doc) {
        HashSet<Integer> X = new HashSet<>();
        String[] words = cur_doc.split("\\s+"); // split words based on whitespaces
        for (int i = 0; i < words.length; i++) {
            words[i] = words[i].replaceAll("\\W", "");
            if (words[i].length() > 0) {
                X.add(getId(words[i]));
            }
        }
        return X;
    }

    private int getId(String word) {
        int id = word.hashCode() % N;
        if (id < 0) id += N;
        return id;
    }

    private boolean labelExists(String[] list, String target) {
        for (String s: list) {
            if (s.equals(target)) {
                return true;
            }
        }
        return false;
    }

    public Object getKeyFromValue(Map hm, Object value) {
        for (Object o : hm.keySet()) {
            if (hm.get(o).equals(value)) {
                return o;
            }
        }
        return null;
    }
}
