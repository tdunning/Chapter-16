package com.tdunning.oscon;

import com.google.common.base.CharMatcher;
import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.io.Resources;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import java.io.IOException;
import java.util.List;

import static org.junit.Assert.assertEquals;


public class DonutEncoderTest {
  private static final Splitter onComma = Splitter.on(",").trimResults(CharMatcher.anyOf("\"\' "));

  @Test
  public void testAddToVector() throws IOException {
    List<String> lines = Resources.readLines(Resources.getResource("donut.csv"), Charsets.UTF_8);
    DonutEncoder encoder = new DonutEncoder(onComma.split(lines.get(0)), ImmutableList.<String>of("x", "y", "shape", "k"));
    Vector v = new RandomAccessSparseVector(40);
    assertEquals(0, v.norm(0), 1e-9);
    encoder.addToVector(onComma.split(lines.get(1)), v);
    assertEquals(6, v.norm(0), 1e-9);
    assertEquals(4.936827227473259, v.norm(1), 1e-9);

    v = new RandomAccessSparseVector(40);
    assertEquals(0, v.norm(0), 1e-9);
    encoder.addToVector(onComma.split(lines.get(2)), v);
    assertEquals(6, v.norm(0), 1e-9);
    assertEquals(5.620153406634927, v.norm(1), 1e-9);
  }

  @Test
  public void testSingleColumn() throws IOException {
    List<String> lines = Resources.readLines(Resources.getResource("donut.csv"), Charsets.UTF_8);
    DonutEncoder encoder = new DonutEncoder(onComma.split(lines.get(0)), ImmutableList.<String>of("y"));
    Vector v = new RandomAccessSparseVector(40);
    assertEquals(0, v.norm(0), 1e-9);
    encoder.addToVector(onComma.split(lines.get(1)), v);
    assertEquals(1, v.norm(0), 1e-9);
    List<String> fields = Lists.newArrayList(onComma.split(lines.get(1)));
    assertEquals(Double.parseDouble(fields.get(1)), v.norm(1), 1e-9);
  }
}
