#include "miopen.hpp"
#include "tensor.hpp"

int main(int argc, char *argv[]) {
  // Regular MIOpen housekeeping
  device_init();
  miopenEnableProfiling(mio::handle(), true);

  Tensor input(16, 128, 16,
               16); // batch size = 16, input channels = 3, image size = 16 x 16
  Tensor output;
  Tensor weights(1, 128, 3, 3); // kernel size = 3 x 3
  Tensor bias;
  miopenConvolutionDescriptor_t conv_desc;

  // initialize tensor
  input.uniform();
  weights.uniform();
  // declarations for fusion
  miopenFusionPlanDescriptor_t fusePlanDesc;
  miopenOperatorArgs_t fusionArgs;
  miopenFusionOpDescriptor_t convoOp;
  miopenFusionOpDescriptor_t biasOp;
  miopenFusionOpDescriptor_t activOp;

  // Create the convolution descriptor
  miopenCreateConvolutionDescriptor(&conv_desc);
  miopenInitConvolutionDescriptor(conv_desc, miopenConvolution, 0, 0, 1, 1, 1,
                                  1);
  // Get the convolution output dimensions
  int n, c, h, w;
  miopenGetConvolutionForwardOutputDim(conv_desc, input.desc, weights.desc, &n,
                                       &c, &h, &w);
  output = Tensor(n, c, h, w);
  bias = Tensor(1, c, 1, 1);
  // Create the fusion plan
  miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, input.desc);
  miopenCreateOperatorArgs(&fusionArgs);
  miopenCreateOpConvForward(fusePlanDesc, &convoOp, conv_desc, weights.desc);

  miopenCreateOpBiasForward(fusePlanDesc, &biasOp, bias.desc);
  // we are only concerned with RELU
  miopenCreateOpActivationForward(fusePlanDesc, &activOp, miopenActivationRELU);

  // compile fusion plan
  auto status = miopenCompileFusionPlan(mio::handle(), fusePlanDesc);
  if (status != miopenStatusSuccess) {
    return -1;
  }
  float alpha = static_cast<float>(1), beta = static_cast<float>(0);
  float activ_alpha = static_cast<float>(0), activ_beta = static_cast<float>(0),
        activ_gamma = static_cast<float>(0);

  // Set the Args
  miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, weights.data);
  miopenSetOpArgsActivForward(fusionArgs, activOp, &alpha, &beta, activ_alpha,
                              activ_beta, activ_gamma);
  miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, bias.data);

  miopenExecuteFusionPlan(mio::handle(), fusePlanDesc, input.desc, input.data,
                          output.desc, output.data, fusionArgs);

  // The same fusion plan may be reused with new arguments without a re-compile
  Tensor input2(
      16, 128, 16,
      16); // batch size = 16, input channels = 3, image size = 16 x 16
  Tensor output2(n, c, h, w);
  Tensor weights2(1, 128, 3, 3); // kernel size = 3 x 3
  Tensor bias2(1, c, 1, 1);

  input2.uniform();
  weights2.uniform();

  // possibly in a loop but with new values for the tensors to be meaningful
  // Here we use the same values to keep the code simple
  for (auto idx = 0; idx < 1; idx++) {
    miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta,
                               weights2.data);
    miopenSetOpArgsActivForward(fusionArgs, activOp, &alpha, &beta, activ_alpha,
                                activ_beta, activ_gamma);
    miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, bias2.data);

    miopenExecuteFusionPlan(mio::handle(), fusePlanDesc, input2.desc,
                            input2.data, output2.desc, output2.data,
                            fusionArgs);
  }

  // Cleanup
  miopenDestroyFusionPlan(fusePlanDesc);
  miopenDestroyConvolutionDescriptor(conv_desc);
}
