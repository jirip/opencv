/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <float.h>

// to make sure we can use these short names
#undef K
#undef L
#undef T

// This is based on the "An Improved Adaptive Background Mixture Model for
// Real-time Tracking with Shadow Detection" by P. KaewTraKulPong and R. Bowden
// http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf
//
// The windowing method is used, but not the shadow detection. I make some of my
// own modifications which make more sense. There are some errors in some of their
// equations.
//

namespace cv
{

BackgroundSubtractor::~BackgroundSubtractor() {}
void BackgroundSubtractor::operator()(InputArray, OutputArray, double)
{
}

//void BackgroundSubtractor::getBackgroundImage(OutputArray) const
//{
//}

static const int defaultNMixtures = 5;
static const int defaultHistory = 200;
static const double defaultBackgroundRatio = 0.7;
static const double defaultVarThreshold = 3.76 * 3.76;//2.5*2.5;
static const double defaultNoiseSigma = 3.;//30*0.5;
static const double defaultInitialWeight = 0.05;

BackgroundSubtractorMOG::BackgroundSubtractorMOG()
{
    frameSize = Size(0,0);
    frameType = 0;

    nframes = 0;
    nmixtures = defaultNMixtures;
    history = defaultHistory;
    varThreshold = defaultVarThreshold;
    backgroundRatio = defaultBackgroundRatio;
    noiseSigma = defaultNoiseSigma;
}

BackgroundSubtractorMOG::BackgroundSubtractorMOG(int _history, int _nmixtures,
                                                 double _backgroundRatio,
                                                 double _noiseSigma)
{
    frameSize = Size(0,0);
    frameType = 0;

    nframes = 0;
    nmixtures = min(_nmixtures > 0 ? _nmixtures : defaultNMixtures, 8);
    history = _history > 0 ? _history : defaultHistory;
    varThreshold = defaultVarThreshold;
    backgroundRatio = min(_backgroundRatio > 0 ? _backgroundRatio : 0.95, 1.);
    noiseSigma = _noiseSigma <= 0 ? defaultNoiseSigma : _noiseSigma;
}

BackgroundSubtractorMOG::~BackgroundSubtractorMOG()
{
}


void BackgroundSubtractorMOG::initialize(Size _frameSize, int _frameType)
{
    frameSize = _frameSize;
    frameType = _frameType;
    nframes = 0;

    int nchannels = CV_MAT_CN(frameType);
    CV_Assert( CV_MAT_DEPTH(frameType) == CV_8U );

    // for each gaussian mixture of each pixel bg model we store ...
    // the mixture sort key (w/sum_of_variances), the mixture weight (w),
    // the mean (nchannels values) and
    // the diagonal covariance matrix (another nchannels values)
    bgmodel.create( 1, frameSize.height*frameSize.width*nmixtures*(3 + 2*nchannels), CV_32F );
    bgmodel = Scalar::all(0);
    bgmodel_o.create( 1, frameSize.height*frameSize.width*nmixtures*(3 + 2*nchannels), CV_32F );
    bgmodel_o = Scalar::all(0);

}

template<typename VT> struct MixData
{
    float sortKey;
    float weight;
	int clusterWeightSum;
    VT mean;
    VT var;
};


static void process_ghost_mask ( const Mat& inmask, Mat& bgmodel, Mat& bgmodel_o, int nmixtures)
{
	int x, y, k, k1, rows = inmask.rows, cols = inmask.cols;
    int K = nmixtures;
    
    MixData<Vec3f>* mptr = (MixData<Vec3f>*)bgmodel.data;
	MixData<Vec3f>* mptr_o = (MixData<Vec3f>*)bgmodel_o.data;
	
    for( y = 0; y < rows; y++ )
    {		
		
		
		const uchar* mask = inmask.ptr<uchar>(y);
		for( x = 0; x < cols; x++, mptr += K , mptr_o += K)
		{
			//std::cout << mptr_o[0].sortKey << std::endl;
			if(mask[x] > 0)
			{
				float w = (float) mask[x] / (float) 255.0;
				float wscale = 1.f/(1 + w);
				
				for( k = 0; k < K; k++ )
				{
					//std::cout << y << ' ' << x << ' ' << k << "y x k" << std::endl;
					
					if(mptr[k].weight > mptr_o[k].weight)
					{
						mptr[k].weight += w;
						break;
					}
				}
				for( k1 = 0; k1 < K; k1++ )
				{
					//std::cout << y << ' ' << x << ' ' << k << "y x k" << std::endl;
					mptr[k1].weight *= wscale;
				}
				for( k1 = k - 1; k1 >= 0; k1-- )
				{
					if( mptr[k1].sortKey <= mptr[k1+1].sortKey )
					{
						std::swap( mptr[k1], mptr[k1+1] );
					}
				}
			}
		}
	}
}

static void process_static_mask ( const Mat& inmask, Mat& bgmodel, Mat& bgmodel_o, int nmixtures)
{
	int x, y, k, k1, rows = inmask.rows, cols = inmask.cols;
    int K = nmixtures;
    
    MixData<Vec3f>* mptr = (MixData<Vec3f>*)bgmodel.data;
	MixData<Vec3f>* mptr_o = (MixData<Vec3f>*)bgmodel_o.data;
	
    for( y = 0; y < rows; y++ )
    {		
		
		
		const uchar* mask = inmask.ptr<uchar>(y);
		for( x = 0; x < cols; x++, mptr += K , mptr_o += K)
		{
			//std::cout << mptr_o[0].sortKey << std::endl;
			if(mask[x] > 0)
			{
				float w = (float) mask[x] / (float) 255.0;
				float ch = 0;
				
				for( k = 0; k < K; k++ )
				{
					//std::cout << y << ' ' << x << ' ' << k << "y x k" << std::endl;
					
					if(mptr[k].weight > mptr_o[k].weight)
					{
						ch = w * (mptr[k].weight - mptr_o[k].weight);
						mptr[k].weight -= ch;
						break;
					}
				}
				for( k1 = 0; k1 < K; k1++ )
				{
					//std::cout << y << ' ' << x << ' ' << k << "y x k" << std::endl;
					mptr[k1].weight *= 1.f/(1 - ch);
				}
				for( k1 = k - 1; k1 >= 0; k1-- )
				{
					if( mptr[k1].sortKey <= mptr[k1+1].sortKey )
					{
						std::swap( mptr[k1], mptr[k1+1] );
					}
				}
			}
		}
	}
}
	

static void process8uC1( const Mat& image, Mat& fgmask, double learningRate,
                         Mat& bgmodel, Mat& bgmodel_o, int nmixtures, double backgroundRatio,
                         double varThreshold, double noiseSigma )
{	
    // Nema zmenu updatu s beta
    
    int x, y, k, k1, rows = image.rows, cols = image.cols;
    float alpha = (float)learningRate, T = (float)backgroundRatio, vT = (float)varThreshold;
    int K = nmixtures;
    
    bgmodel.copyTo(bgmodel_o);
    MixData<float>* mptr = (MixData<float>*)bgmodel.data;

    const float w0 = (float)defaultInitialWeight;
    const float sk0 = (float)(w0/(defaultNoiseSigma*2));
    const float var0 = (float)(defaultNoiseSigma*defaultNoiseSigma*4);
    const float minVar = (float)(noiseSigma*noiseSigma);

    for( y = 0; y < rows; y++ )
    {
        const uchar* src = image.ptr<uchar>(y);
        uchar* dst = fgmask.ptr<uchar>(y);

        if( alpha > 0 )
        {
            for( x = 0; x < cols; x++, mptr += K )
            {
                float wsum = 0;
                float pix = src[x];
                int kHit = -1, kForeground = -1;

                for( k = 0; k < K; k++ )
                {
                    float w = mptr[k].weight;
                    wsum += w;
                    if( w < FLT_EPSILON )
                        break;
                    float mu = mptr[k].mean;
                    float var = mptr[k].var;
                    float diff = pix - mu;
                    float d2 = diff*diff;
                    if( d2 < vT*var )
                    {
                        wsum -= w;
                        float dw = alpha*(1.f - w);
                        mptr[k].weight = w + dw;
                        mptr[k].mean = mu + alpha*diff;
                        var = max(var + alpha*(d2 - var), minVar);
                        mptr[k].var = var;
                        mptr[k].sortKey = w/sqrt(var);

                        for( k1 = k-1; k1 >= 0; k1-- )
                        {
                            if( mptr[k1].sortKey >= mptr[k1+1].sortKey )
                                break;
                            std::swap( mptr[k1], mptr[k1+1] );
                        }

                        kHit = k1+1;
                        break;
                    }
                }

                if( kHit < 0 ) // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
                {
                    kHit = k = min(k, K-1);
                    wsum += w0 - mptr[k].weight;
                    mptr[k].weight = w0;
                    mptr[k].mean = pix;
                    mptr[k].var = var0;
                    mptr[k].sortKey = sk0;
                }
                else
                    for( ; k < K; k++ )
                        wsum += mptr[k].weight;

                float wscale = 1.f/wsum;
                wsum = 0;
                for( k = 0; k < K; k++ )
                {
                    wsum += mptr[k].weight *= wscale;
                    mptr[k].sortKey *= wscale;
                    if( wsum > T && kForeground < 0 )
                        kForeground = k+1;
                }

                dst[x] = (uchar)(-(kHit >= kForeground));
            }
        }
        else
        {
            for( x = 0; x < cols; x++, mptr += K )
            {
                float pix = src[x];
                int kHit = -1, kForeground = -1;

                for( k = 0; k < K; k++ )
                {
                    if( mptr[k].weight < FLT_EPSILON )
                        break;
                    float mu = mptr[k].mean;
                    float var = mptr[k].var;
                    float diff = pix - mu;
                    float d2 = diff*diff;
                    if( d2 < vT*var )
                    {
                        kHit = k;
                        break;
                    }
                }

                if( kHit >= 0 )
                {
                    float wsum = 0;
                    for( k = 0; k < K; k++ )
                    {
                        wsum += mptr[k].weight;
                        if( wsum > T )
                        {
                            kForeground = k+1;
                            break;
                        }
                    }
                }

                dst[x] = (uchar)(kHit < 0 || kHit >= kForeground ? 255 : 0);
            }
        }
    }
}


static void process8uC3( const Mat& image, Mat& fgmask, double learningRate,
                         Mat& bgmodel, Mat& bgmodel_o, int nmixtures, double backgroundRatio,
                         double varThreshold, double noiseSigma, int nframes)
{
    int x, y, k, k1, rows = image.rows, cols = image.cols;
    float alpha = (float)learningRate, T = (float)backgroundRatio, vT = (float)varThreshold;
    int K = nmixtures;

    float beta = (1 - alpha); // / alpha;
    alpha = max(alpha, (float) 1./nframes);

    const float w0 = (float)defaultInitialWeight;
    const float sk0 = (float)(w0/(defaultNoiseSigma*2*sqrt(3.)));
    const float var0 = (float)(defaultNoiseSigma*defaultNoiseSigma*4);
    const float minVar = (float)(noiseSigma*noiseSigma);
    
    bgmodel.copyTo(bgmodel_o);
    MixData<Vec3f>* mptr = (MixData<Vec3f>*)bgmodel.data;

    for( y = 0; y < rows; y++ )
    {
        const uchar* src = image.ptr<uchar>(y);
        uchar* dst = fgmask.ptr<uchar>(y);

        if( alpha > 0 )
        {
            for( x = 0; x < cols; x++, mptr += K )
            {
                float wsum = 0;
                Vec3f pix(src[x*3], src[x*3+1], src[x*3+2]);
                int kHit = -1, kForeground = -1;

                for( k = 0; k < K; k++ )
                {
                    float w = mptr[k].weight;
                    wsum += w;
                    if( w < FLT_EPSILON )
                        break;
                    Vec3f mu = mptr[k].mean;
                    Vec3f var = mptr[k].var;
                    Vec3f diff = pix - mu;
                    Vec3f diff2;
                    divide(diff, var, diff2);
                    float d2 = diff.dot(diff2);

                    if( d2 < vT )
                    {
                        wsum -= w;
                        
                        float prevClusterWeightSumBeta = mptr[k].clusterWeightSum * beta;
                        float newClusterWeightSum = prevClusterWeightSumBeta + 1.f;
                        
                        mptr[k].clusterWeightSum = newClusterWeightSum;
                        
                        float dw = alpha*(1.f - w);
                        mptr[k].weight = w + dw;
                         
                        mptr[k].mean = (mu * prevClusterWeightSumBeta + pix) / newClusterWeightSum;
                        var = Vec3f(max((var[0] * prevClusterWeightSumBeta + diff[0]*diff[0]) / newClusterWeightSum, minVar*3),
                                    max((var[1] * prevClusterWeightSumBeta + diff[1]*diff[1]) / newClusterWeightSum, minVar),
                                    max((var[2] * prevClusterWeightSumBeta + diff[2]*diff[2]) / newClusterWeightSum, minVar));
                         
                        mptr[k].var = var;
                        mptr[k].sortKey = w/sqrt(var[0] + var[1] + var[2]);

                        for( k1 = k-1; k1 >= 0; k1-- )
                        {
                            if( mptr[k1].sortKey >= mptr[k1+1].sortKey )
                                break;
                            std::swap( mptr[k1], mptr[k1+1] );
                        }

                        kHit = k1+1;
                        break;
                    }
                }

                if( kHit < 0 ) // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
                {
                    kHit = k = min(k, K-1);
                    wsum += w0 - mptr[k].weight;
                    mptr[k].weight = w0;
                    mptr[k].clusterWeightSum = 1;
                    mptr[k].mean = pix;
                    mptr[k].var = Vec3f(var0*3, var0, var0);
                    mptr[k].sortKey = sk0;
                }
                else
                    for( ; k < K; k++ )
                        wsum += mptr[k].weight;

                float wscale = 1.f/wsum;
                wsum = 0;
                for( k = 0; k < K; k++ )
                {
                    wsum += mptr[k].weight *= wscale;
                    mptr[k].sortKey *= wscale;
                    if( wsum > T && kForeground < 0 )
                        kForeground = k+1;
                }

                dst[x] = (uchar)(-(kHit >= kForeground));
            }
        }
        else
        {
            // Bez uceni, neupdatovane
            for( x = 0; x < cols; x++, mptr += K )
            {
                Vec3f pix(src[x*3], src[x*3+1], src[x*3+2]);
                int kHit = -1, kForeground = -1;

                for( k = 0; k < K; k++ )
                {
                    if( mptr[k].weight < FLT_EPSILON )
                        break;
                    Vec3f mu = mptr[k].mean;
                    Vec3f var = mptr[k].var;
                    Vec3f diff = pix - mu;
                    float d2 = diff.dot(diff);
                    if( d2 < vT*(var[0] + var[1] + var[2]) )
                    {
                        kHit = k;
                        break;
                    }
                }

                if( kHit >= 0 )
                {
                    float wsum = 0;
                    for( k = 0; k < K; k++ )
                    {
                        wsum += mptr[k].weight;
                        if( wsum > T )
                        {
                            kForeground = k+1;
                            break;
                        }
                    }
                }

                dst[x] = (uchar)(kHit < 0 || kHit >= kForeground ? 255 : 0);
            }
        }
    }
}

void BackgroundSubtractorMOG::operator()(InputArray _image, OutputArray _fgmask, double learningRate)
{
    Mat image = _image.getMat();
    bool needToInitialize = nframes == 0 || learningRate >= 1 || image.size() != frameSize || image.type() != frameType;

    if( needToInitialize )
        initialize(image.size(), image.type());

    CV_Assert( image.depth() == CV_8U );
    _fgmask.create( image.size(), CV_8U );
    Mat fgmask = _fgmask.getMat();

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./history; // 1./min(nframes, history)
    CV_Assert(learningRate >= 0);

    if( image.type() == CV_8UC1 )
        process8uC1( image, fgmask, learningRate, bgmodel, bgmodel_o, nmixtures, backgroundRatio, varThreshold, noiseSigma);
    else if( image.type() == CV_8UC3 )
        process8uC3( image, fgmask, learningRate, bgmodel, bgmodel_o, nmixtures, backgroundRatio, varThreshold, noiseSigma, nframes);
    else
        CV_Error( CV_StsUnsupportedFormat, "Only 1- and 3-channel 8-bit images are supported in BackgroundSubtractorMOG" );
}

void BackgroundSubtractorMOG::ghost_mask(InputArray _inmask)
{
	Mat inmask = _inmask.getMat();
	process_ghost_mask(inmask, bgmodel, bgmodel_o, nmixtures);

}

void BackgroundSubtractorMOG::static_mask(InputArray _inmask)
{
	Mat inmask = _inmask.getMat();
	process_static_mask(inmask, bgmodel, bgmodel_o, nmixtures);

}

void BackgroundSubtractorMOG::getBackgroundImageMean(OutputArray backgroundImage) const
{
    int x, y, k, K = nmixtures;
    int nchannels = CV_MAT_CN(frameType);
    CV_Assert(nchannels == 1 || nchannels == 3);

    Mat meanBackground(frameSize, CV_MAKETYPE(CV_8U, nchannels), Scalar::all(0));
    MixData<Vec3f>* mptr = (MixData<Vec3f>*)bgmodel.data;

    std::vector<float> meanVal(nchannels, 0.f);
    for( y = 0; y < frameSize.height; y++ )
    {
        for( x = 0; x < frameSize.width; x++, mptr += K )
        {
            float totalWeight = 0.f;
            for( k = 0; k < K; k++ )
            {   
                float w = mptr[k].weight;
                
                for(int chn = 0; chn < nchannels; chn++)
                {
                    meanVal[chn] += w * mptr[k].mean[chn];
                }
                
                totalWeight += w;

                if(totalWeight > backgroundRatio)
                    break;
            }
            float invWeight = 1.f/totalWeight;
            switch(nchannels)
            {
            case 1:
                meanBackground.at<uchar>(y, x) = (uchar)(meanVal[0] * invWeight);
                meanVal[0] = 0.f;
                break;
            case 3:
                Vec3f& meanVec = *reinterpret_cast<Vec3f*>(&meanVal[0]);
                meanBackground.at<Vec3b>(y, x) = Vec3b(meanVec * invWeight);
                meanVec = 0.f;
                break;
            }
        }
        
    }
    meanBackground.copyTo(backgroundImage);
}

// Required for Ghost detection Evaluators
void BackgroundSubtractorMOG::getBackgroundImage(OutputArray backgroundImage) const
{
    int x, y, k, K = nmixtures;
    int nchannels = CV_MAT_CN(frameType);
    CV_Assert(nchannels == 1 || nchannels == 3);

    Mat meanBackground(frameSize, CV_MAKETYPE(CV_8U, nchannels), Scalar::all(0));
    MixData<Vec3f>* mptr = (MixData<Vec3f>*)bgmodel.data;

    std::vector<float> meanVal(nchannels, 0.f);
    for( y = 0; y < frameSize.height; y++ )
    {
        for( x = 0; x < frameSize.width; x++, mptr += K )
        {
            k = 0;
			for(int chn = 0; chn < nchannels; chn++)
			{
				meanVal[chn] +=  mptr[k].mean[chn];
			}
			
            switch(nchannels)
            {
            case 1:
                meanBackground.at<uchar>(y, x) = (uchar)(meanVal[0]);
                meanVal[0] = 0.f;
                break;
            case 3:
                Vec3f& meanVec = *reinterpret_cast<Vec3f*>(&meanVal[0]);
                meanBackground.at<Vec3b>(y, x) = Vec3b(meanVec);
                meanVec = 0.f;
                break;
            }
        }
        
    }
    meanBackground.copyTo(backgroundImage);
}

void BackgroundSubtractorMOG::getBackgroundImgs(OutputArray backgroundImage0, OutputArray backgroundImage1, OutputArray backgroundImage2, OutputArray activeBackground) const
{
    int x, y, k, K = nmixtures;
    int nchannels = CV_MAT_CN(frameType);
    CV_Assert(nchannels == 1 || nchannels == 3);

    Mat background0(frameSize, CV_MAKETYPE(CV_8U, nchannels), Scalar::all(0));
    Mat background1(frameSize, CV_MAKETYPE(CV_8U, nchannels), Scalar::all(0));
    Mat background2(frameSize, CV_MAKETYPE(CV_8U, nchannels), Scalar::all(0));
    Mat active(frameSize, CV_MAKETYPE(CV_8U, 1), Scalar::all(0));
    MixData<Vec3f>* mptr = (MixData<Vec3f>*)bgmodel.data;

    std::vector<float> meanVal(nchannels, 0.f);
    for( y = 0; y < frameSize.height; y++ )
    {
        for( x = 0; x < frameSize.width; x++, mptr += K )
        {
            int ks = -1;
            // First 
            k = 0;
            for(int chn = 0; chn < nchannels; chn++)
			{
				meanVal[chn] =  mptr[k].mean[chn];
			}
            
            switch(nchannels)
            {
            case 1:
                background0.at<uchar>(y, x) = (uchar)(meanVal[0]);
                meanVal[0] = 0.f;
            case 3:
                Vec3f& meanVec = *reinterpret_cast<Vec3f*>(&meanVal[0]);
                background0.at<Vec3b>(y, x) = Vec3b(meanVec);
                meanVec[0] = 0.f;
                
            }
            
            float totalWeight = mptr[k].weight;
            if(totalWeight > backgroundRatio)
            {
                active.at<uchar>(y, x) = (uchar)(255);
                ks = 0;
                }
            
            // Second
            k = 1;
            
			for(int chn = 0; chn < nchannels; chn++)
			{
				meanVal[chn] =  mptr[k].mean[chn];
			}
            switch(nchannels)
            {
            case 1:
                background1.at<uchar>(y, x) = (uchar)(meanVal[0]);
                meanVal[0] = 0.f;
            case 3:
                Vec3f& meanVec = *reinterpret_cast<Vec3f*>(&meanVal[0]);
                background1.at<Vec3b>(y, x) = Vec3b(meanVec);
                meanVec[0] = 0.f;
                
            }
            
            totalWeight += mptr[k].weight;
            if(totalWeight > backgroundRatio && ks < 0)
            {
                active.at<uchar>(y, x) = (uchar)(180);
                ks = 1;
                }
            // Third
            k = 2;
			for(int chn = 0; chn < nchannels; chn++)
			{
				meanVal[chn] =  mptr[k].mean[chn];
			}
            switch(nchannels)
            {
            case 1:
                background2.at<uchar>(y, x) = (uchar)(meanVal[0]);
                meanVal[0] = 0.f;
            case 3:
                Vec3f& meanVec = *reinterpret_cast<Vec3f*>(&meanVal[0]);
                background2.at<Vec3b>(y, x) = Vec3b(meanVec);
                meanVec[0] = 0.f;
                
            }
            totalWeight += mptr[k].weight;
            if(totalWeight > backgroundRatio && ks < 0)
            {
                active.at<uchar>(y, x) = (uchar)(90);
                ks = 2;
                }
            
            if(ks < 0)
            {
                active.at<uchar>(y, x) = (uchar)(0);
                }
        }
        
    }
    background0.copyTo(backgroundImage0);
    background1.copyTo(backgroundImage1);
    background2.copyTo(backgroundImage2);
    active.copyTo(activeBackground);
}

}

/* End of file. */
