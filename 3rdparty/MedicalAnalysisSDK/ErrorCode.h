/*!
 * \file ErrorCode.h
 * \date 2017/05/09 16:28
 *
 * \author blue
 * Contact: yang.wang@shangyiyun.com
 *
 * \brief 错误状态码
 *
 * TODO: long description
 *
 * \note
*/

#pragma once

namespace SYY {
	enum ErrorCode
	{
		SYY_NO_ERROR = 0,					// 返回成功

		// SDK Init
		SYY_SDK_REPEAT_INIT = 1,			// 重复初始化sdk
		SYY_SDK_NO_INIT = 2,				// 没有初始化sdk

		// ERROR
		SYY_SYS_ERROR = 10,					// 系统错误
		SYY_LOG_INIT_NO_PERMISSION = 11,	// 日志初始化失败，程序无权限
		SYY_NO_IMPLEMENTION = 11,			// 算法未实现
	};
}

