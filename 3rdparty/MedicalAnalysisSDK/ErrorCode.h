/*!
 * \file ErrorCode.h
 * \date 2017/05/09 16:28
 *
 * \author blue
 * Contact: yang.wang@shangyiyun.com
 *
 * \brief ����״̬��
 *
 * TODO: long description
 *
 * \note
*/

#pragma once

namespace SYY {
	enum ErrorCode
	{
		SYY_NO_ERROR = 0,					// ���سɹ�

		// SDK Init
		SYY_SDK_REPEAT_INIT = 1,			// �ظ���ʼ��sdk
		SYY_SDK_NO_INIT = 2,				// û�г�ʼ��sdk

		// ERROR
		SYY_SYS_ERROR = 10,					// ϵͳ����
		SYY_LOG_INIT_NO_PERMISSION = 11,	// ��־��ʼ��ʧ�ܣ�������Ȩ��
		SYY_NO_IMPLEMENTION = 11,			// �㷨δʵ��
	};
}

