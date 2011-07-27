package com.tdunning.oscon;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ContinuousValueEncoder;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Encodes donut data.  Assumes that we are passed column values that correspond to the column
 * titles we were given to start.
 * <p/>
 * We will encode some subset of the variables as specified at construction time. <ul> <li>x, y, a,
 * b, c - continuous position variables</li> <li>k - cluster id, k-means with target</li> <li>k0 -
 * cluster id, k-means without target</li> <li>shape - symbol shape</li> </ul>
 */
public class DonutEncoder {
  private static final Map<String, FeatureVectorEncoder> encoderMap;
  static {
    ImmutableMap.Builder<String, FeatureVectorEncoder> builder = ImmutableMap.builder();
    encoderMap = builder.put("x", new ContinuousValueEncoder("x"))
      .put("y", new ContinuousValueEncoder("y"))
      .put("a", new ContinuousValueEncoder("a"))
      .put("b", new ContinuousValueEncoder("b"))
      .put("c", new ContinuousValueEncoder("c"))
      .put("xx", new ContinuousValueEncoder("xx"))
      .put("xy", new ContinuousValueEncoder("xy"))
      .put("yy", new ContinuousValueEncoder("yy"))
      .put("k", new StaticWordValueEncoder("k"))
      .put("k0", new StaticWordValueEncoder("k0"))
      .put("shape", new StaticWordValueEncoder("shape"))
      .put("bias", new ContinuousValueEncoder("bias"))
      .build();
  }

  private int[] columns;
  private List<String> names;

  public DonutEncoder(Iterable<String> names, Iterable<String> encodeThese) {
    Set<String> encodable = ImmutableSet.copyOf(encodeThese);
    columns = new int[encodable.size()];
    int i = 0;
    int column = 0;
    this.names = Lists.newArrayList(names);
    for (String name : this.names) {
      if (encodable.contains(name)) {
        columns[i++] = column;
      }
      column++;
    }
  }

  public void addToVector(Iterable<String> values, Vector v) {
    int i = 0;
    int column = 0;
    for (String value : values) {
      if (i >= columns.length) {
        break;
      }
      if (columns[i] == column) {
        encoderMap.get(names.get(column)).addToVector(value, v);
        i++;
      }
      column++;
    }
  }

  public int[] getColumns() {
    return columns;
  }
}
