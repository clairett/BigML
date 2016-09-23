import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.*;


import java.io.*;
import java.util.*;

/**
 * Created by tiantian on 9/11/16.
 */
public class NB_train_hadoop {
    public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(LongWritable key, Text value, OutputCollector<Text, IntWritable> context, Reporter reporter) throws IOException {
            BufferedReader br = new BufferedReader(new StringReader(value.toString()));
            String l;
            while ((l = br.readLine()) != null && l.length() != 0) {
                String[] line = l.split("\t");
                String[] labels = line[1].split(",");
                Vector<String> tokens = tokenizeDoc(line[2]);
                Iterator it = tokens.iterator();
                for (String label : labels) {
                    word.set("Y="+label);
                    context.collect(word, one);

                    while (it.hasNext()) {
                        word.set("Y="+label+",W="+it.next());
                        context.collect(word, one);
                    }
                    word.set("Y=" + label + ",W=*");
                    context.collect(word, new IntWritable(tokens.size()));
                }
                word.set("Y=*");
                context.collect(word, new IntWritable(labels.length));
            }
            br.close();
        }
    }

    public static class Reduce extends MapReduceBase implements Reducer<Text, IntWritable, Text, IntWritable> {
        public void reduce(Text key, Iterator<IntWritable> values, OutputCollector<Text, IntWritable> context, Reporter reporter) throws IOException {
            int sum = 0;
            while (values.hasNext()) {
                sum += values.next().get();
            }
            context.collect(key, new IntWritable(sum));
        }
    }
    static Vector<String> tokenizeDoc(String cur_doc) {
        String[] words = cur_doc.split("\\s+"); // split words based on whitespaces
        Vector<String> tokens = new Vector<String>();
        for (int i = 0; i < words.length; i++) {
            words[i] = words[i].replaceAll("\\W", "");
            if (words[i].length() > 0) {
                tokens.add(words[i]);
            }
        }
        return tokens;
    }
}
