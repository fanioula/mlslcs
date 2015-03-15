package gr.auth.ee.lcs.implementations.global;

import java.io.IOException;

public class AverageEvaluationAcrossTens {

	public static void main(String[] args) {
		
		int repetitions = 10;
		
		try {
			for (int i = 0; i < repetitions; i++) {
//				AllMlTypes newAllMlTypes = (AllMlTypes) Class.forName("gr.auth.ee.lcs.implementations.global.AllMlTypes").newInstance();
				AllMlTypes.main(args);
			}
		} catch (InstantiationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
